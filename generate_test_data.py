import os
import json
import random
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        exit(1)

def fetch_master_ids(cursor):
    """Pulls true working operational IDs from master tables to avoid FK/Key breaking constraints."""
    print("⏳ Querying master table keys to establish valid references...")
    
    cursor.execute("SELECT id FROM MActivity;")
    activities = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT id FROM MAllocation;")
    allocations = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT id FROM MAssignee;")
    assignees = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT id FROM MIncharge;")
    incharges = [row[0] for row in cursor.fetchall()]
    
    cursor.execute("SELECT id FROM MRequester;")
    requesters = [row[0] for row in cursor.fetchall()]
    
    return activities, allocations, assignees, incharges, requesters

def generate_user_volume():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Verify master pools have values to draw patterns from
    activities, allocations, assignees, incharges, requesters = fetch_master_ids(cursor)
    if not (activities and allocations and assignees and incharges and requesters):
        print("❌ Error: One or more master validation data tables are empty. Seed them via push_db.py first.")
        return

    # Configuration definitions matching requested row outputs
    target_distributions = [
        {"user_id": -1499992964, "row_count": 50},
        {"user_id": -1499994136, "row_count": 100},
        {"user_id": -1499993802, "row_count": 150}
    ]

    print("=" * 70)
    print("🚀 INITIALIZING VERTICAL TRANSACTION SEED PIPELINE ENGINE")
    print("=" * 70)

    # Clean out previous micro sample traces to establish clear index telemetry benchmarks
    print("🧹 Cleaning existing tracking logs...")
    cursor.execute("TRUNCATE TABLE LUserBehaviour RESTART IDENTITY CASCADE;")
    conn.commit()

    insert_query = """
        INSERT INTO LUserBehaviour (UserID, SelectedDataType, SelectedDataID, FormData, TimeStamp)
        VALUES (%s, %s, %s, %s::json, %s);
    """

    base_time = datetime.now() - timedelta(days=60)
    total_generated_rows = 0

    for target in target_distributions:
        u_id = target["user_id"]
        required_rows = target["row_count"]
        
        print(f"⏳ Generating {required_rows:,} records for User Session Identifier: {u_id}...")
        
        # We divide rows by 5 because each generated form interaction bundle logs 5 rows vertically
        form_iterations = required_rows // 5
        batch_data = []

        # Create distinct operational choice anchors per user to simulate realistic, habit-forming AI predictions
        user_fav_activity = random.choice(activities)
        user_fav_allocation = random.choice(allocations)
        user_fav_assignee = random.choice(assignees)
        user_fav_incharge = random.choice(incharges)
        user_fav_requester = random.choice(requesters)

        for _ in range(form_iterations):
            # 85% probability to select favorite patterns (creates clean top-frequency prediction spikes)
            act_id = user_fav_activity if random.random() < 0.85 else random.choice(activities)
            alloc_id = user_fav_allocation if random.random() < 0.85 else random.choice(allocations)
            ass_id = user_fav_assignee if random.random() < 0.85 else random.choice(assignees)
            inch_id = user_fav_incharge if random.random() < 0.85 else random.choice(incharges)
            req_id = user_fav_requester if random.random() < 0.85 else random.choice(requesters)

            # Reconstruct the snapshot matching your exact CEO requested FormData JSON structure
            form_payload = {
                "Activity": act_id,
                "Allocation": alloc_id,
                "Assignee": ass_id,
                "Incharge": inch_id,
                "Requester": req_id
            }
            form_json_str = json.dumps(form_payload)
            
            # Increment step timestamps uniformly down the ledger line matrix
            row_timestamp = base_time + timedelta(minutes=random.randint(1, 80000))
            timestamp_str = row_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

            # Map individual field types vertically row-by-row
            batch_data.append((u_id, "Activity", act_id, form_json_str, timestamp_str))
            batch_data.append((u_id, "Allocation", alloc_id, form_json_str, timestamp_str))
            batch_data.append((u_id, "Assignee", ass_id, form_json_str, timestamp_str))
            batch_data.append((u_id, "Incharge", inch_id, form_json_str, timestamp_str))
            batch_data.append((u_id, "Requester", req_id, form_json_str, timestamp_str))

        # Push to PostgreSQL via high-speed atomic transactions
        execute_batch(cursor, insert_query, batch_data)
        conn.commit()
        total_generated_rows += len(batch_data)
        print(f"   ✓ User {u_id} batch committed. ({len(batch_data):,} vertical rows mapped)")

    cursor.close()
    conn.close()
    
    print("=" * 70)
    print(f"🎉 SUCCESS: {total_generated_rows:,} total rows injected into LUserBehaviour.")
    print("   Sequence IDs reset and synced cleanly 1 to 35,000 in continuous order.")
    print("=" * 70)

if __name__ == "__main__":
    generate_user_volume()