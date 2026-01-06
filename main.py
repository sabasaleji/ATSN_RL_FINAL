# main.py
"""
MAIN ORCHESTRATOR

Flow:
1. Decide topic + post type
2. Generate prompts (via generate.py + RL)
3. Store content (post_contents)
4. Queue for scheduled posting
5. Collect metrics (via cron job)
6. Compute reward
7. Update RL
"""

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()

# from db import fetch_or_calculate_reward

import uuid
import time
import random
from datetime import datetime
import numpy as np
import pytz

# Indian Standard Time (IST) - Asia/Kolkata
IST = pytz.timezone("Asia/Kolkata")
#from campaign import topic,date,time,platform

import db
# from rl_agent import update_rl
from generate import generate_prompts,embed_topic,generate_topic
from job_queue import queue_reward_calculation_job
from content_generation import generate_content

# Add imports
from post_scheduler import run_scheduling_job, run_posting_job
import time
from datetime import datetime, timedelta

# Add this function
def check_and_run_scheduled_jobs():
    """Check current time and run scheduled jobs if needed"""
    current_time = datetime.now(IST)
    current_time_str = current_time.strftime("%H:%M")
    
    # Run scheduling job at 8:30 AM IST daily
    if current_time_str == "0:30":
        print("🌅 Time is 7:30 AM - Running scheduling job...")
        try:
            run_scheduling_job()
            print("✅ Scheduling job completed")
        except Exception as e:
            print(f"❌ Scheduling job failed: {e}")

    # Run posting jobs at scheduled times
    posting_times = ["03:30", "08:00", "13:30", "16:30"]  # Morning, Afternoon, Evening, Night
    
    if current_time_str in posting_times:
        print(f"🚀 Time is {current_time_str} - Running posting job...")
        try:
            run_posting_job()
            print("✅ Posting job completed")
        except Exception as e:
            print(f"❌ Posting job failed: {e}")


# MAIN LOOP
# -------------------------------------------------

def run_one_post(BUSINESS_ID, platform, time=None, day_of_week=None):
    # Get user's scheduling preferences if not provided
    if time is None or day_of_week is None:
        scheduling_prefs = db.get_profile_scheduling_prefs(BUSINESS_ID)
        time = time or scheduling_prefs["time_bucket"]
        day_of_week = day_of_week if day_of_week is not None else scheduling_prefs["day_of_week"]

    print(f"\n🚀 Starting new post cycle for {platform} at {time} (day {day_of_week})")
    
    date = datetime.now(IST).date().isoformat()


    # ---------- 1️⃣ BUSINESS CONTEXT ----------

    # Get business embedding and profile data from profiles table
    business_embedding = db.get_profile_embedding_with_fallback(BUSINESS_ID)
    if business_embedding is None:
        raise RuntimeError(f"No business embedding found for business {BUSINESS_ID}. Business profile must be created first.")

    profile_data = db.get_profile_business_data(BUSINESS_ID)

    #generate topic
    topic_data = generate_topic(
    business_context=str(profile_data),
    platform=platform,
    date=date)
    
    topic_text = topic_data["topic"]

    #create embedding for topic
    topic_embedding = embed_topic(topic_text)

    print("🧵 Topic:", topic_text)
    print("🧠 Topic embedding dim:", len(topic_embedding))



    # ---------- 2️⃣ GENERATE PROMPTS (RL INSIDE) ----------
    inputs = {
        "BUSINESS_AESTHETIC": profile_data["brand_voice"],  # Use brand voice as aesthetic
        "BUSINESS_TYPES": profile_data["business_types"],
        "INDUSTRIES": profile_data["industries"],
        "BUSINESS_DESCRIPTION": profile_data["business_description"],
    }

    result = generate_prompts(
        inputs,
        business_embedding,
        topic_embedding,
        platform,
        time,
        day_of_week,
        topic_text,profile_data,
        business_context=profile_data,
    )

    # Extract values based on mode
    action = result["action"]
    context = result["context"]
    ctx_vec = result["ctx_vec"]
    mode = result["mode"]
    prompt_text = result.get("grok_prompt") or result.get("prompt", "") or result.get("image_prompt", "")

    # ---------- 3️⃣ STORE RL ACTION ----------
    post_id = f"{platform}_{uuid.uuid4().hex[:8]}"

    action_id = db.insert_action(
        post_id=post_id,
        platform=platform,
        context=context,
        action=action
    )

    # ---------- 4️⃣ STORE POST CONTENT ----------
    # Extract prompts based on mode (handle both trendy and standard modes)
    image_prompt = result.get("image_prompt",
        f"Create an image with {action['VISUAL_STYLE']} style, {action['TONE']} tone, {action['CREATIVITY']} creativity level.The topic is {topic_text}. Make it engaging for {platform}.Do not include caption in the image  directly.just learn from the caption and generate the image.")

    caption_prompt = result.get("caption_prompt",
        f"Write a {action['TONE']} caption in {action['LENGTH']} length with {action['CREATIVITY']} creativity level. The topic is {topic_text}. Make it suitable for {platform}.")

    print("🎨 Generating caption and image content...")
    content_result = generate_content(caption_prompt, image_prompt, profile_data)

    if content_result["status"] == "success":
        generated_caption = content_result["caption"]
        generated_image_url = content_result["image_url"]

        # Replace {{CAPTION}} placeholder in image_prompt with actual generated caption
        if generated_caption and "{{CAPTION}}" in image_prompt:
            image_prompt = image_prompt.replace("{{CAPTION}}", generated_caption)
            print("🔄 Updated image prompt with generated caption")

        print("✅ Content generated successfully and stored")
        print(f"📝 Caption: {generated_caption[:100]}...")

    else:
        error_msg = f"Content generation failed: {content_result['error']}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    db.insert_post_content(
        post_id=post_id,
        action_id=action_id,
        platform=platform,
        business_id=BUSINESS_ID,
        topic=topic_text,
        # business_context= profile_data["business_description"],
        # business_aesthetic=profile_data["brand_voice"],
        image_prompt=image_prompt,
        caption_prompt=caption_prompt,
        generated_caption=generated_caption,
        generated_image_url=generated_image_url
    )

    # ---------- 5️⃣ QUEUE FOR SCHEDULING ----------


    # Create initial reward record for future calculation
    db.create_post_reward_record(BUSINESS_ID, post_id, platform, action_id)

    # ---------- 6️⃣ QUEUE REWARD CALCULATION FOR WORKER ----------
    # Queue reward calculation job (will automatically trigger RL update when ready)
    job_id = queue_reward_calculation_job(BUSINESS_ID, post_id, platform)
    print(f"📋 Reward calculation queued for worker processing (job: {job_id})")

    print("✅ Post cycle completed - RL learning will happen asynchronously")

# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
ALLOWED_PLATFORMS = {"instagram","facebook"}

if __name__ == "__main__":
    check_and_run_scheduled_jobs()
    try:
        # Get all business profiles
        all_business_ids = db.get_all_profile_ids()
        print(f"📊 Found {len(all_business_ids)} business profiles to check")

        # Process each business
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

                        if platform not in ALLOWED_PLATFORMS:
                            print(f"❌ Skipping unsupported platform: {platform} for business {business_id}")
                            continue  # skip unsupported platforms

                        print(f"🚀 Creating post for business {business_id} on {platform}")
                        
                        run_one_post(
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
        print(f"❌ Critical error in main process: {e}")
        raise
