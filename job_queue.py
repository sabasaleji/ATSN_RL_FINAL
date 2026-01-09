# job_queue.py - Simple job system for RL learning

import asyncio
import time
import threading
from queue import Queue
from typing import Dict, Any, Optional
from datetime import datetime
import pytz

# Indian Standard Time (IST) - Asia/Kolkata
IST = pytz.timezone("Asia/Kolkata")
import db
import rl_agent
import snaphot_collector
# Note: check_and_run_scheduled_jobs is imported dynamically to avoid circular imports

# Thread-safe job queue (replace with Redis/Celery for production)
job_queue = Queue()
job_results = {}  # Store job results by job_id
running_jobs = set()  # Track running job IDs

class Job:
    def __init__(self, job_type: str, job_id: str, payload: Dict[str, Any]):
        self.job_type = job_type  # "reward_calculation" or "rl_update"
        self.job_id = job_id
        self.payload = payload
        self.created_at = datetime.now(IST)
        self.status = "queued"

async def process_reward_calculation_job(job: Job) -> Dict[str, Any]:
    """Process reward calculation job"""
    try:
        payload = job.payload
        profile_id = payload["profile_id"]
        post_id = payload["post_id"]
        platform = payload["platform"]

        print(f"Processing reward calculation for {post_id} on {platform}")

        # Calculate reward
        result = db.fetch_or_calculate_reward(profile_id, post_id, platform)

        # Debug: Print result
        print(f"🔍 Reward calculation result: {result}")

        if result["status"] == "calculated":
            # Queue RL update job
            rl_job = Job(
                job_type="rl_update",
                job_id=f"rl_{post_id}_{int(time.time())}",
                payload={
                    "profile_id": profile_id,
                    "post_id": post_id,
                    "platform": platform,
                    "reward_value": result["reward"]
                }
            )
            job_queue.put(rl_job)
            print(f"📋 Queued RL update job for {post_id}")

        return result

    except Exception as e:
        print(f"❌ Error in reward calculation job: {e}")
        return {"status": "error", "error": str(e)}

async def process_rl_update_job(job: Job) -> Dict[str, Any]:
    """Process RL update job"""
    try:
        payload = job.payload
        profile_id = payload["profile_id"]
        post_id = payload["post_id"]
        platform = payload["platform"]
        reward_value = payload["reward_value"]

        print(f"🧠 Processing RL update for {post_id} (reward: {reward_value:.4f})")

        # Get action and context from database
        # This assumes the action and context are stored during posting
        action_data = get_action_and_context_from_db(post_id, platform, profile_id)

        if not action_data:
            print(f"⚠️  No action data found for {post_id}, skipping RL update")
            return {"status": "skipped", "reason": "no_action_data"}

        action = action_data["action"]
        context = action_data["context"]
        ctx_vec = action_data["ctx_vec"]

        # Get current baseline using pure mathematical update
        current_baseline = db.update_baseline_mathematical(platform, reward_value, beta=0.1)

        # Update RL
        rl_agent.update_rl(
            context=context,
            action=action,
            ctx_vec=ctx_vec,
            reward=reward_value,
            baseline=current_baseline
        )

        print(f"✅ RL update completed for {post_id}")
        return {"status": "completed", "baseline": current_baseline}

    except Exception as e:
        print(f"❌ Error in RL update job: {e}")
        return {"status": "error", "error": str(e)}

def get_action_and_context_from_db(post_id: str, platform: str, profile_id: str) -> Optional[Dict[str, Any]]:
    """Get action and context data from database for RL update"""
    try:
        # Get action data from rl_actions table
        action_result = db.supabase.table("rl_actions").select("*").eq("post_id", post_id).eq("platform", platform).execute()

        if not action_result.data:
            return None

        action_row = action_result.data[0]

        # Reconstruct action dict
        action = {
            "HOOK_TYPE": action_row.get("hook_type"),
            "INFORMATION_DEPTH": action_row.get("information_depth"),
            "TONE": action_row.get("tone"),
            "CREATIVITY": action_row.get("creativity"),
            "COMPOSITION_STYLE": action_row.get("composition_style"),
            "VISUAL_STYLE": action_row.get("visual_style")
        }

        # Get the topic from the post data
        topic = action_row.get("topic", "")
        topic_embedding = db.embed_topic(topic) if topic else None

        # Get business embedding
        business_embedding = db.get_profile_embedding_with_fallback(profile_id)
        if business_embedding is None:
            print(f"❌ No business embedding found for {profile_id}, cannot perform RL update")
            return None

        # Use topic embedding if available, otherwise use business embedding
        final_topic_embedding = topic_embedding if topic_embedding is not None else business_embedding

        # Reconstruct context with real business data
        context = {
            "platform": platform,
            "time_bucket": action_row.get("time_bucket"),
            "business_embedding": business_embedding,
            "topic_embedding": final_topic_embedding
        }

        # Reconstruct context vector
        from rl_agent import build_context_vector
        ctx_vec = build_context_vector(context)

        return {
            "action": action,
            "context": context,
            "ctx_vec": ctx_vec
        }

    except Exception as e:
        print(f"❌ Error retrieving action data for {post_id}: {e}")
        return None

def job_worker():
    """Main job processing worker (synchronous)"""
    print("🚀 Starting RL job worker...")

    while True:
        try:
            print("Job worker waiting for jobs...")
            job = job_queue.get()  # Blocking get
            print(f"📥 Job worker received job: {job.job_id}")
            job.status = "running"
            running_jobs.add(job.job_id)

            print(f"📋 Processing job {job.job_id} ({job.job_type})")

            # Create new event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                if job.job_type == "reward_calculation":
                    result = loop.run_until_complete(process_reward_calculation_job(job))
                elif job.job_type == "rl_update":
                    result = loop.run_until_complete(process_rl_update_job(job))
                else:
                    result = {"status": "error", "error": f"Unknown job type: {job.job_type}"}

                # Debug: Print job completion
                print(f"✅ Job {job.job_id} completed with result: {result}")

                job_results[job.job_id] = result
            finally:
                loop.close()

            running_jobs.remove(job.job_id)

        except Exception as e:
            print(f"❌ Job worker error: {e}")
            time.sleep(1)  # Brief pause on error

def queue_reward_calculation_job(profile_id: str, post_id: str, platform: str) -> str:
    """Queue a reward calculation job"""
    job_id = f"reward_{post_id}_{int(time.time())}"
    job = Job(
        job_type="reward_calculation",
        job_id=job_id,
        payload={
            "profile_id": profile_id,
            "post_id": post_id,
            "platform": platform
        }
    )

    job_queue.put(job)
    print(f"📋 Queued reward calculation job: {job_id}")
    print(f"📊 Current queue size: {job_queue.qsize()}")
    return job_id

async def run_job_worker_async():
    """Async wrapper for the synchronous job worker"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, job_worker)


async def run_unified_service():
    """Run both job worker and metrics collection concurrently"""
    print("🚀 Starting concurrent job processing and metrics collection...")

    # Create tasks for both services
    job_task = asyncio.create_task(run_job_worker_async())
    metrics_task = asyncio.create_task(snaphot_collector.run_continuous_metrics_collection())

    # Wait for both to complete (they run indefinitely until interrupted)
    await asyncio.gather(job_task, metrics_task, return_exceptions=True)


if __name__ == "__main__":
    """Run unified service (jobs + metrics collection)"""
    print("🔄 Starting Unified RL Service (Jobs + Metrics)...")
    print("📋 This will run job processing and metrics collection continuously")
    print("⚠️  Use Ctrl+C to stop gracefully")

    try:
        # Content generation and reward processing
        

        # Check if it's 11pm UTC and run main.py if so (independent of scheduled jobs check)
        current_utc = datetime.now(pytz.UTC)
        if current_utc.hour == 6:  # 6am UTC
            print("🌙 It's 6am UTC - Running main.py...")
            try:
                # Import main module functions dynamically
                import main

                # Get all business profiles and process them (without calling check_and_run_scheduled_jobs again)
                all_business_ids = db.get_all_profile_ids()
                print(f"📊 Found {len(all_business_ids)} business profiles to check")

                # Process each business (this replicates main.py's main loop)
                for business_id in all_business_ids:
                    try:
                        print(f"\n🏢 Processing business: {business_id}")

                        # Check if this business should create posts today
                        if not db.should_create_post_today(business_id):
                            print(f"⏸️ Skipping business {business_id} — not scheduled for today (IST)")
                            continue

                        # Get connected platforms for this business
                        user_connected_platforms = list(set(db.get_connected_platforms(business_id)))
                        print(f"📱 Business {business_id} has {len(user_connected_platforms)} connected platforms: {user_connected_platforms}")

                        # Create posts for each platform
                        for platform in user_connected_platforms:
                            try:
                                platform = platform.lower().strip()  # normalize

                                if platform not in main.ALLOWED_PLATFORMS:
                                    print(f"❌ Skipping unsupported platform: {platform} for business {business_id}")
                                    continue  # skip unsupported platforms

                                print(f"🚀 Creating post for business {business_id} on {platform}")

                                main.run_one_post(
                                    BUSINESS_ID=business_id,
                                    platform=platform,
                                )
                                print(f"✅ Successfully processed post for {business_id} on {platform}")

                            except Exception as e:
                                print(f"❌ Failed to create post for {business_id} on {platform}: {e}")
                                continue  # Continue with other platforms

                    except Exception as e:
                        print(f"❌ Failed to process business {business_id}: {e}")
                        continue  # Continue with other businesses

                print("\n✅ Daily post creation process completed for all businesses")

            except Exception as e:
                print(f"❌ Error running main.py at 11pm UTC: {e}")

        # Run the unified service (both job worker and metrics collection)
        asyncio.run(run_unified_service())

    except KeyboardInterrupt:
        print("\n🛑 Unified service stopped by user")
    except Exception as e:
        print(f"❌ Unified service error: {e}")
        raise






