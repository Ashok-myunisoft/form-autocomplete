import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load Environment Database Target Configs
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# --- MASTER EMPLOYEE DIRECTORY DATA (FROM MLOGIN) ---
EMPLOYEES = [
    {"id": -1499999938, "name": "VV"},
    {"id": -1499999077, "name": "Venkatesh Nataraj"},
    {"id": -1499994137, "name": "Naveenasri"},
    {"id": -1499994136, "name": "Visa P"},
    {"id": -1499994135, "name": "Pooja Prakash"},
    {"id": -1499994133, "name": "Joswa"},
    {"id": -1499994107, "name": "Maha Vishnu M"},
    {"id": -1499994080, "name": "Durga J"},
    {"id": -1499994079, "name": "Indhuja N"},
    {"id": -1499994076, "name": "Kanishka K"},
    {"id": -1499994074, "name": "Sankar Ganesh R"},
    {"id": -1499994071, "name": "Sabitha Shree"},
    {"id": -1499993940, "name": "Arun M"},
    {"id": -1499993802, "name": "Ashok R"},
    {"id": -1499993728, "name": "Kaviya M"},
    {"id": -1499993673, "name": "Vatchala G"},
    {"id": -1499993635, "name": "Gogulnath D"},
    {"id": -1499993615, "name": "Mohammed Muzaffar K"},
    {"id": -1499993581, "name": "Thilai Kumar P"},
    {"id": -1499993524, "name": "Shri Kiruthiga"},
    {"id": -1499993500, "name": "Durgeswari K"},
    {"id": -1499992993, "name": "Vishnu Shankar B"},
    {"id": -1499992975, "name": "Dharun Palanisamy"},
    {"id": -1499992973, "name": "Akalya Krishnan"},
    {"id": -1499992970, "name": "Prathibha L"},
    {"id": -1499992969, "name": "Yoganathan"},
    {"id": -1499992964, "name": "Akshaia Sai Diwakar"},
    {"id": -1499992962, "name": "Ganesan A"},
    {"id": -1499992961, "name": "Manivel S"},
    {"id": -1499992959, "name": "PRITIKA G"},
    {"id": -1499992958, "name": "Sanchana N M"},
    {"id": -1499992956, "name": "Himavarsini K"}
]

def generate_behavioral_matrix():
    print("Connecting to PostgreSQL Infrastructure...")
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    cur = conn.cursor()

    # 1. DYNAMICALLY FETCH GENUINE MASTER RECORDS FROM DATABASE TABLES
    print("Fetching active master records from configuration tables...")
    
    cur.execute("SELECT id, name FROM MActivity;")
    db_activities = cur.fetchall()
    
    cur.execute("SELECT id, name FROM MAllocation;")
    db_allocations = cur.fetchall()
    
    cur.execute("SELECT id, name FROM MAssignee;")
    db_assignees = cur.fetchall()
    
    cur.execute("SELECT id, name FROM MIncharge;")
    db_incharges = cur.fetchall()
    
    cur.execute("SELECT id, name FROM MRequester;")
    db_requesters = cur.fetchall()

    # Safety validation check to make sure tables aren't completely empty
    if not db_activities or not db_allocations or not db_assignees or not db_incharges or not db_requesters:
        print("❌ CRITICAL ERROR: One or more master tables (MActivity, MAllocation, MAssignee, MIncharge, MRequester) contain zero records.")
        print("Please ensure your master tables are seeded before running this behavioral generator.")
        return

    # 2. CLEAR PREVIOUS BEHAVIOR LOGS
    print("Clearing historical LUserBehaviour tracking records...")
    cur.execute("TRUNCATE TABLE LUserBehaviour RESTART IDENTITY;")
    conn.commit()

    print(f"Beginning dynamic dataset generation for {len(EMPLOYEES)} profiles...")
    
    total_inserted_rows = 0
    base_time = datetime.now() - timedelta(days=15) 
    execution_sequence = ["Activity", "Allocation", "Assignee", "Incharge", "Requester"]

    for emp in EMPLOYEES:
        user_id = emp["id"]
        
        # Pick a pool of favorite habits out of real master data options for this specific user
        favored_activities = random.sample(db_activities, min(3, len(db_activities)))
        favored_allocations = random.sample(db_allocations, min(3, len(db_allocations)))
        favored_assignees = random.sample(db_assignees, min(2, len(db_assignees)))
        favored_incharges = random.sample(db_incharges, min(2, len(db_incharges)))
        favored_requesters = random.sample(db_requesters, min(2, len(db_requesters)))

        # Generate 8 form sessions * 5 clicks per session = 40 records per user
        for session_idx in range(8):
            # Form state starts fresh for this single session sequence
            form_state = {
                "Activity": None, "Allocation": None, "Assignee": None, "Incharge": None, "Requester": None
            }
            
            # Map out a complete, high-confidence unified pattern combination for this form pass
            current_session_pattern = {
                "Activity": random.choice(favored_activities),
                "Allocation": random.choice(favored_allocations),
                "Assignee": random.choice(favored_assignees),
                "Incharge": random.choice(favored_incharges),
                "Requester": random.choice(favored_requesters)
            }
            
            base_time += timedelta(minutes=random.randint(15, 120))

            # Simulate step-by-step clicks across the 5 input boxes rows
            for click_step, field_key in enumerate(execution_sequence):
                target_master_node = current_session_pattern[field_key]
                selected_data_id = int(target_master_node["id"])
                display_data = str(target_master_node["name"])
                
                # Append updated input data to current snapshot state mapping
                form_state[field_key] = selected_data_id
                json_form_data = json.dumps(form_state)
                
                insert_query = """
                    INSERT INTO LUserBehaviour (UserID, FormID, SelectedDataType, SelectedDataID, DisplayData, FormData, TimeStamp)
                    VALUES (%s, %s, %s, %s, %s, %s::json, %s);
                """
                
                cur.execute(insert_query, (
                    user_id,
                    1, 
                    field_key,
                    selected_data_id,
                    display_data,
                    json_form_data,
                    base_time + timedelta(seconds=click_step * 3) 
                ))
                total_inserted_rows += 1

        print(f"   ✓ Generated 40 sequential valid database logs for user: {emp['name']}")
        conn.commit()

    cur.close()
    conn.close()
    
    print("=" * 60)
    print(f"🎉 SUCCESS: Dynamic seed completed. {total_inserted_rows} rows pushed.")
    print("=" * 60)

if __name__ == "__main__":
    generate_behavioral_matrix()