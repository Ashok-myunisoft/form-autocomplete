import os
import json
import base64
import gzip
import requests
import concurrent.futures
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_CONNECTION_STRING") or os.getenv("DB_URL")

DB_POOL: Optional[ThreadedConnectionPool] = None

MODEL_CACHE = {}
MASTER_DATA_CACHE = {}

LAST_LOGIN_DTO: Optional[Dict[str, Any]] = None


def get_login_dto(login: Optional[str] = Header(None)) -> Dict[str, Any]:
    global LAST_LOGIN_DTO

    if login:
        try:
            parsed = json.loads(login)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid 'login' header: it must be a JSON-encoded object.",
            )
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid 'login' header: JSON must decode to an object.",
            )
        LAST_LOGIN_DTO = parsed
        return parsed

    if LAST_LOGIN_DTO is not None:
        return LAST_LOGIN_DTO

    raise HTTPException(
        status_code=400,
        detail="Missing 'login' header. Send it at least once (on any endpoint) before omitting it.",
    )


SVC_TO_ENTITY_NAME = {
    "requester": "Requester",
    "activity": "Activity",
    "allocation": "Allocation",
    "assignee": "Assignee",
    "incharge": "Incharge",
    "task": "Task",
}

def get_config_payload():
    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing configuration metadata context file: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)

def _fetch_one_svc_dropdown(field: str, url: str, login_dto: dict):
    session = requests.Session()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "login": json.dumps(login_dto)
    }
    try:
        response = session.post(url, data=b"", headers=headers, timeout=12)
        if response.status_code != 200:
            return field, []

        response_json = response.json()
        raw_body_string = (response_json.get("Body") or "").strip()
        if not raw_body_string:
            return field, []

        clean_base64_str = raw_body_string.strip('"').strip()
        compressed_bytes = base64.b64decode(clean_base64_str)
        decompressed_bytes = gzip.decompress(compressed_bytes)
        clean_json_text = decompressed_bytes.decode("utf-8").strip()

        unpacked_data = json.loads(clean_json_text)
        inner_body = unpacked_data.get("Body", "")

        if not isinstance(inner_body, str) or not inner_body:
            return field, []

        raw_rows = json.loads(inner_body)
        clean_list = []
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            item_id = item.get("Id") or item.get("id") or item.get("PK_Id") or item.get("Value")
            item_name = item.get("Name") or item.get("name") or item.get("Text") or item.get("Description")
            if item_id is not None and item_name:
                clean_list.append({"id": item_id, "name": str(item_name).strip()})

        return field, clean_list
    except Exception as e:
        print(f"❌ SVC [{field}] connection loop execution error: {e}")
        return field, []

def sync_master_data_on_demand(login_dto: dict, user_space_key: str):
    if user_space_key in MASTER_DATA_CACHE:
        return True

    norm_dto = {k.lower(): v for k, v in login_dto.items()}
    base_url = norm_dto.get("baseurl") or login_dto.get("BaseURL") or ""
    base_url = str(base_url).strip().rstrip("/")
    if not base_url or base_url == "None":
        print("❌ Dynamic Setup Failure: loginDTO lacks a valid environment BaseURL parameter context.")
        return False

    config = get_config_payload()
    endpoints_map = config.get("SVC_ENDPOINTS", {})

    # Build into LOCAL dicts first, publish atomically at the end - avoids
    # a concurrent request for the same user seeing a half-populated cache.
    local_master_data = {}
    local_model_cache = {}

    valid_endpoints = {
        key: suffix for key, suffix in endpoints_map.items()
        if SVC_TO_ENTITY_NAME.get(key)
    }

    # Fetch all SVC dropdown endpoints in parallel rather than one-by-one.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(valid_endpoints), 1)) as executor:
        future_to_key = {
            executor.submit(
                _fetch_one_svc_dropdown, key, f"{base_url}/{suffix.lstrip('/')}", login_dto
            ): key
            for key, suffix in valid_endpoints.items()
        }
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            entity = SVC_TO_ENTITY_NAME[key]
            try:
                _, rows = future.result()
            except Exception as e:
                print(f"❌ SVC [{key}] connection loop execution error: {e}")
                rows = []

            local_master_data[entity] = rows

            if rows:
                names = [row["name"] for row in rows]
                vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
                tfidf_matrix = vectorizer.fit_transform(names)
                local_model_cache[entity.lower()] = {
                    "vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix, "names": names
                }

    MASTER_DATA_CACHE[user_space_key] = local_master_data
    MODEL_CACHE[user_space_key] = local_model_cache
    return True

# 🔑 Replaces ALL of the previous header/Security/"Authorize" machinery.
# loginDTO now arrives via the "login" header (see get_login_dto above,
# which also handles the "send it once, reuse afterwards" fallback).
# This function pulls UserId out of it and lazily syncs master data - no
# separate auth endpoint, no Swagger padlock, nothing to configure.
def resolve_login_context(login_dto: Dict[str, Any]) -> dict:
    if not login_dto:
        raise HTTPException(status_code=400, detail="login_dto is required (via the 'login' header).")

    norm_dto = {k.lower(): v for k, v in login_dto.items()}
    user_id = norm_dto.get("userid")
    if user_id is None:
        raise HTTPException(status_code=400, detail="login_dto is missing a UserId field.")

    db_name = norm_dto.get("databasename", "default_db")
    user_space_key = f"{user_id}_{db_name}"

    success = sync_master_data_on_demand(login_dto, user_space_key)
    if not success:
        raise HTTPException(status_code=400, detail="login_dto lacks a valid BaseURL field for master data sync.")

    return {"user_id": user_id, "user_space_key": user_space_key}

app = FastAPI(title="Dynamic Autocomplete Engine")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
def startup_db_pool():
    global DB_POOL
    DB_POOL = ThreadedConnectionPool(5, 50, DATABASE_URL, cursor_factory=RealDictCursor)

@app.on_event("shutdown")
def shutdown_db_pool():
    if DB_POOL:
        DB_POOL.closeall()

def get_db():
    conn = DB_POOL.getconn()
    try:
        yield conn
    finally:
        DB_POOL.putconn(conn)

class SingleTransactionPayload(BaseModel):
    selected_datatype: str
    selected_dataid: int
    display_data: str
    formdata_snapshot: Dict[str, Optional[int]]

class SequentialFormSubmission(BaseModel):
    form_id: int = 1
    records: List[SingleTransactionPayload]

class AutocompleteContextRequest(BaseModel):
    form_id: int = 1
    entity_type: str
    query: str
    current_form_state: Dict[str, Any]

class PredictionMatrixRequest(BaseModel):
    form_id: int = 1
    trigger_datatype: str
    trigger_dataid: int

@app.post("/gbaiapi/predict-remaining-fields")
def predict_remaining_fields_context(
    payload: PredictionMatrixRequest,
    conn=Depends(get_db),
    login_dto: Dict[str, Any] = Depends(get_login_dto),
):
    session = resolve_login_context(login_dto)
    user_id = session["user_id"]
    user_space_key = session["user_space_key"]
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT formdata FROM luserbehaviour 
                WHERE userid = %s AND formid = %s AND formdata IS NOT NULL AND formdata->>%s = %s
                ORDER BY 
                (
                    CASE WHEN formdata->>'Activity' IS NOT NULL AND formdata->>'Activity' <> 'null' THEN 1 ELSE 0 END +
                    CASE WHEN formdata->>'Allocation' IS NOT NULL AND formdata->>'Allocation' <> 'null' THEN 1 ELSE 0 END +
                    CASE WHEN formdata->>'Assignee' IS NOT NULL AND formdata->>'Assignee' <> 'null' THEN 1 ELSE 0 END +
                    CASE WHEN formdata->>'Incharge' IS NOT NULL AND formdata->>'Incharge' <> 'null' THEN 1 ELSE 0 END +
                    CASE WHEN formdata->>'Requester' IS NOT NULL AND formdata->>'Requester' <> 'null' THEN 1 ELSE 0 END
                ) DESC,
                count DESC,
                timestamp DESC,
                userbehaviourid DESC
                LIMIT 1;
            """, (user_id, payload.form_id, payload.trigger_datatype, str(payload.trigger_dataid)))
            row = cur.fetchone()
            if not row or not row["formdata"]:
                return {"predictions": {}}

            historic_form_state = row["formdata"]
            user_master = MASTER_DATA_CACHE.get(user_space_key, {})
            resolved_predictions = {}
            normalized_master = {k.lower(): v for k, v in user_master.items()}
            
            for datatype, dataid_val in historic_form_state.items():
                dt_lower = datatype.lower()
                if dt_lower == payload.trigger_datatype.lower() or dataid_val is None:
                    continue
                try:
                    target_id = int(dataid_val)
                    lookup_key = "incharge" if dt_lower == "incharge" else dt_lower
                    master_rows = normalized_master.get(lookup_key, [])
                    matched_node = next((item for item in master_rows if int(item["id"]) == target_id), None)
                    if matched_node:
                        display_key = next((k for k in user_master.keys() if k.lower() == dt_lower), datatype)
                        if display_key.lower() == "incharge":
                            display_key = "Incharge"
                        resolved_predictions[display_key] = {
                            "id": target_id,
                            "name": matched_node["name"]
                        }
                except Exception as ex:
                    print(f"Parsing error on key [{datatype}]: {ex}")
                    continue
            return {"predictions": resolved_predictions}
    except Exception as e:
        print(f"❌ Matrix prediction error: {e}")
        return {"predictions": {}}

@app.post("/gbaiapi/save-form")
def save_form_data(
    payload: SequentialFormSubmission,
    conn=Depends(get_db),
    login_dto: Dict[str, Any] = Depends(get_login_dto),
):
    session = resolve_login_context(login_dto)
    user_id = session["user_id"]

    try:
        with conn.cursor() as cur:
            for record in payload.records:
                json_form_data = json.dumps(record.formdata_snapshot)

                # UPDATE-first, INSERT-only-on-miss keeps userbehaviourid
                # gapless: unchanged combinations just get count += 1 with
                # no identity sequence value consumed; only a genuinely new
                # (userid, formid, selecteddatatype, selecteddataid) combo
                # falls through to the INSERT branch below.
                cur.execute("""
                    UPDATE luserbehaviour
                    SET count = count + 1,
                        formdata = %s::jsonb,
                        displaydata = %s,
                        timestamp = NOW()
                    WHERE userid = %s
                    AND formid = %s
                    AND selecteddatatype = %s
                    AND selecteddataid = %s;
                """, (
                    json_form_data,
                    record.display_data.strip(),
                    user_id,
                    payload.form_id,
                    record.selected_datatype,
                    record.selected_dataid,
                ))

                if cur.rowcount == 0:
                    cur.execute("""
                        INSERT INTO luserbehaviour
                        (userid, formid, selecteddatatype, selecteddataid, displaydata, formdata, count, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, 1, NOW());
                    """, (
                        user_id,
                        payload.form_id,
                        record.selected_datatype,
                        record.selected_dataid,
                        record.display_data.strip(),
                        json_form_data,
                    ))

        conn.commit()
        return {"status": "success", "message": "All fields successfully saved using progressive snapshots."}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gbaiapi/autocomplete")
def get_contextual_predictions(
    payload: AutocompleteContextRequest,
    conn=Depends(get_db),
    login_dto: Dict[str, Any] = Depends(get_login_dto),
):
    session = resolve_login_context(login_dto)
    user_id = session["user_id"]
    user_space_key = session["user_space_key"]
    top_historical_id = None
    context_dict = {}
    
    for k, v in payload.current_form_state.items():
        if k == payload.entity_type or v is None:
            continue
        try: context_dict[k] = int(v)
        except (ValueError, TypeError): continue

    with conn.cursor() as cur:
        if context_dict:
            json_match_string = json.dumps(context_dict)
            query = """
                SELECT selecteddataid
                FROM luserbehaviour
                WHERE userid = %s AND formid = %s AND selecteddatatype = %s
                  AND formdata @> %s::jsonb
                ORDER BY count DESC, timestamp DESC, userbehaviourid DESC
                LIMIT 1;
            """
            cur.execute(query, [user_id, payload.form_id, payload.entity_type, json_match_string])
            row = cur.fetchone()
            if row:
                top_historical_id = int(row["selecteddataid"])

        if top_historical_id is None:
            cur.execute("""
                SELECT selecteddataid as selected_id
                FROM luserbehaviour
                WHERE userid = %s AND formid = %s AND selecteddatatype = %s
                GROUP BY selecteddataid
                ORDER BY SUM(count) DESC, MAX("timestamp") DESC, MAX(userbehaviourid) DESC
                LIMIT 1;
            """, (user_id, payload.form_id, payload.entity_type))
            row = cur.fetchone()
            if row: top_historical_id = int(row["selected_id"])

    query_str = payload.query.strip().lower()
    user_master = MASTER_DATA_CACHE.get(user_space_key, {})
    master_items = user_master.get(payload.entity_type, [])
    if not master_items: return []

    final_response = []
    ai_item = None
    regular_items = []

    entity_key = payload.entity_type.lower()
    user_models = MODEL_CACHE.get(user_space_key, {})
    similarities = {}
    if query_str and entity_key in user_models:
        try:
            artifact = user_models[entity_key]
            query_vec = artifact["vectorizer"].transform([query_str])
            raw_scores = cosine_similarity(query_vec, artifact["tfidf_matrix"]).flatten()
            similarities = {name.lower(): score for name, score in zip(artifact["names"], raw_scores)}
        except Exception: pass

    for item in master_items:
        item_id = int(item["id"])
        item_name = item["name"]

        if query_str and query_str not in item_name.lower(): continue

        is_ai = (top_historical_id is not None and item_id == top_historical_id)
        score = similarities.get(item_name.lower(), 0.0) if query_str else 0.0

        dropdown_node = {
            "id": item_id, "code": item.get("code"), "name": item_name,
            "display_text": f"{item_id}. {item_name}", "is_ai_prediction": is_ai, "score": score
        }
        if is_ai: ai_item = dropdown_node
        else: regular_items.append(dropdown_node)

    if query_str: regular_items.sort(key=lambda x: x["score"], reverse=True)
    if ai_item is not None:
        ai_item.pop("score", None)
        final_response.append(ai_item)
    for item in regular_items:
        item.pop("score", None)
        final_response.append(item)

    return final_response