# metrics_collector.py - Collect social media metrics and store in post_snapshots
"""
Social Media Metrics Collector

This module collects engagement metrics from social media platforms at specific time intervals
after posts are published. It stores the data in the post_snapshots table for reward calculation.

Collection intervals: 6, 24, 48, 72, and 168 hours after posting
"""

import os
import asyncio
import httpx
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from cryptography.fernet import Fernet
import logging
import pytz

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Indian Standard Time (IST) - Asia/Kolkata
IST = pytz.timezone("Asia/Kolkata")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import db

# Collection intervals in hours (matching REWARD_WEIGHTS)
COLLECTION_INTERVALS = [6, 24, 48, 72, 168]


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt encrypted access token"""
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if not encryption_key:
        return encrypted_token  # Fallback if not encrypted

    fernet = Fernet(encryption_key.encode())
    return fernet.decrypt(encrypted_token.encode()).decode()


def get_platform_credentials(platform: str, business_id: str) -> Optional[Dict[str, str]]:
    """Get platform access token and page ID for a business"""
    try:
        res = db.supabase.table("platform_connections") \
            .select("access_token_encrypted, page_id, page_username") \
            .eq("user_id", business_id) \
            .eq("platform", platform) \
            .eq("is_active", True) \
            .eq("connection_status", "active") \
            .execute()

        if res.data and len(res.data) > 0:
            connection = res.data[0]
            access_token = decrypt_token(connection.get("access_token_encrypted", ""))
            page_id = connection.get("page_id")

            if access_token and page_id:
                return {
                    'access_token': access_token,
                    'page_id': page_id
                }

        # Fallback to environment variables for development
        env_token_key = f"{platform.upper()}_ACCESS_TOKEN"
        env_page_key = f"{platform.upper()}_PAGE_ID"
        access_token = os.getenv(env_token_key)
        page_id = os.getenv(env_page_key)

        if access_token and page_id:
            return {
                'access_token': access_token,
                'page_id': page_id
            }

        return None

    except Exception as e:
        logger.error(f"Error getting platform credentials for {business_id} on {platform}: {e}")
        return None


async def fetch_facebook_follower_count(page_id: str, access_token: str) -> int:
    """Fetch Facebook Page follower count"""
    try:
        async with httpx.AsyncClient() as client:
            url = f"https://graph.facebook.com/v18.0/{page_id}"

            params = {
                "access_token": access_token,
                "fields": "followers_count"
            }

            response = await client.get(url, params=params)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch Facebook follower count: {response.text}")
                return 0

            data = response.json()
            return data.get("followers_count", 0)

    except Exception as e:
        logger.error(f"Error fetching Facebook follower count: {e}")
        return 0


async def fetch_instagram_follower_count(instagram_account_id: str, access_token: str) -> int:
    """Fetch Instagram Business Account follower count"""
    try:
        async with httpx.AsyncClient() as client:
            url = f"https://graph.facebook.com/v18.0/{instagram_account_id}"

            params = {
                "access_token": access_token,
                "fields": "followers_count"
            }

            response = await client.get(url, params=params)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch Instagram follower count: {response.text}")
                return 0

            data = response.json()
            return data.get("followers_count", 0)

    except Exception as e:
        logger.error(f"Error fetching Instagram follower count: {e}")
        return 0


async def fetch_facebook_post_metrics(page_id: str, access_token: str, media_id: str) -> Dict[str, Any]:
    """
    Fetch metrics for a Facebook post using the Graph API

    Args:
        page_id: Facebook Page ID
        access_token: Page access token
        media_id: Post ID to fetch metrics for

    Returns:
        Dictionary containing post metrics
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"https://graph.facebook.com/v18.0/{media_id}/insights"

            params = {
                "access_token": access_token,
                "metric": "post_impressions,post_engaged_users,post_reactions_by_type_total,post_comments,post_shares"
            }

            logger.info(f"📊 Fetching Facebook metrics for post {media_id}")
            response = await client.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"Facebook API error: {response.status_code} - {response.text}")
                return {"error": f"Failed to fetch metrics: {response.text}"}

            data = response.json()
            logger.info(f"✅ Successfully fetched Facebook metrics for post {media_id}")

            # Fetch follower count
            follower_count = await fetch_facebook_follower_count(page_id, access_token)

            # Parse insights data
            metrics = {
                "impressions": 0,
                "reach": 0,
                "engaged_users": 0,
                "likes": 0,
                "comments": 0,
                "shares": 0,
                "reactions": 0,
                "follower_count": follower_count
            }

            if "data" in data:
                for insight in data["data"]:
                    metric_name = insight["name"]
                    if "values" in insight and len(insight["values"]) > 0:
                        value = insight["values"][0].get("value", 0)

                        if metric_name == "post_impressions":
                            metrics["impressions"] = int(value) if isinstance(value, (int, str)) and str(value).isdigit() else 0
                        elif metric_name == "post_engaged_users":
                            metrics["engaged_users"] = int(value) if isinstance(value, (int, str)) and str(value).isdigit() else 0
                        elif metric_name == "post_reactions_by_type_total":
                            # Sum all reaction types
                            if isinstance(value, dict):
                                total_reactions = sum(int(count) for count in value.values() if str(count).isdigit())
                                metrics["reactions"] = total_reactions
                                # Likes are part of reactions, but we'll count them separately if available
                                metrics["likes"] = value.get("like", 0)
                        elif metric_name == "post_comments":
                            metrics["comments"] = int(value) if isinstance(value, (int, str)) and str(value).isdigit() else 0
                        elif metric_name == "post_shares":
                            metrics["shares"] = int(value) if isinstance(value, (int, str)) and str(value).isdigit() else 0

            return metrics

    except Exception as e:
        logger.error(f"Error fetching Facebook post metrics: {e}")
        return {"error": str(e)}


async def fetch_instagram_post_metrics(instagram_account_id: str, access_token: str, media_id: str) -> Dict[str, Any]:
    """
    Fetch metrics for an Instagram post using the Graph API

    Args:
        instagram_account_id: Instagram Business Account ID
        access_token: Access token with instagram_manage_insights permission
        media_id: Media ID to fetch metrics for

    Returns:
        Dictionary containing post metrics
    """
    try:
        async with httpx.AsyncClient() as client:
            url = f"https://graph.facebook.com/v18.0/{media_id}/insights"

            params = {
                "access_token": access_token,
                "metric": "likes,comments,saved,total_interactions"
            }

            logger.info(f"📊 Fetching Instagram metrics for media {media_id}")
            response = await client.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"Instagram API error: {response.status_code} - {response.text}")
                return {"error": f"Failed to fetch metrics: {response.text}"}

            data = response.json()
            logger.info(f"✅ Successfully fetched Instagram metrics for media {media_id}")

            # Fetch follower count
            follower_count = await fetch_instagram_follower_count(instagram_account_id, access_token)

            # Parse insights data
            metrics = {
                "impressions": 0,  # Not available for individual media in newer API versions
                "reach": 0,
                "engagement": 0,  # Will be set from total_interactions
                "likes": 0,
                "comments": 0,
                "replies": 0,
                "saves": 0,
                "shares": 0,  # Instagram doesn't have shares, but we'll include for consistency
                "follower_count": follower_count
            }

            if "data" in data:
                for insight in data["data"]:
                    metric_name = insight["name"]
                    if "values" in insight and len(insight["values"]) > 0:
                        value = insight["values"][0].get("value", 0)

                        if metric_name == "total_interactions":
                            metrics["engagement"] = int(value) if isinstance(value, (int, str)) and str(value).isdigit() else 0
                        elif metric_name == "likes":
                            metrics["likes"] = int(value) if isinstance(value, (int, str)) and str(value).isdigit() else 0
                        elif metric_name == "comments":
                            metrics["comments"] = int(value) if isinstance(value, (int, str)) and str(value).isdigit() else 0
                        elif metric_name == "saved":
                            metrics["saves"] = int(value) if isinstance(value, (int, str)) and str(value).isdigit() else 0

            return metrics

    except Exception as e:
        logger.error(f"Error fetching Instagram post metrics: {e}")
        return {"error": str(e)}


def get_recently_posted_content(hours_threshold: int = 24) -> List[Dict[str, Any]]:
    """
    Get posts that have been posted and need metrics collection

    Args:
        hours_threshold: Only look at posts from the last N hours to avoid processing old posts

    Returns:
        List of posts with media_id that need metrics collection
    """
    try:
        # Calculate cutoff time
        cutoff_time = datetime.now(IST) - timedelta(hours=hours_threshold)

        # Query for posted content with media_id
        res = db.supabase.table("post_contents") \
            .select("post_id, platform, business_id, media_id, created_at") \
            .eq("status", "posted") \
            .neq("media_id", None) \
            .gte("created_at", cutoff_time.isoformat()) \
            .execute()

        posts = []
        if res.data:
            for row in res.data:
                posts.append({
                    "post_id": row["post_id"],
                    "platform": row["platform"],
                    "business_id": row["business_id"],
                    "media_id": row["media_id"],
                    "created_at": row["created_at"]
                })

        logger.info(f"📝 Found {len(posts)} recently posted content items with media_id")
        return posts

    except Exception as e:
        logger.error(f"Error fetching recently posted content: {e}")
        return []


def calculate_collection_times(post_created_at: str) -> List[Dict[str, int]]:
    """
    Calculate when metrics should be collected based on post creation time

    Args:
        post_created_at: ISO format datetime string when post was created

    Returns:
        List of dicts with hours_since_post and whether collection is due
    """
    try:
        # Parse post creation time
        if post_created_at.endswith('Z'):
            post_time = datetime.fromisoformat(post_created_at[:-1])
        else:
            post_time = datetime.fromisoformat(post_created_at)

        # Ensure timezone awareness
        if post_time.tzinfo is None:
            post_time = IST.localize(post_time)
        elif post_time.tzinfo != IST:
            post_time = post_time.astimezone(IST)

        current_time = datetime.now(IST)
        time_diff = current_time - post_time
        hours_since_post = time_diff.total_seconds() / 3600

        collection_times = []
        for interval_hours in COLLECTION_INTERVALS:
            time_until_collection = interval_hours - hours_since_post

            collection_times.append({
                "hours": interval_hours,
                "hours_since_post": hours_since_post,
                "time_until_collection": time_until_collection,
                "is_due": time_until_collection <= 0  # Due if we've passed the collection time
            })

        return collection_times

    except Exception as e:
        logger.error(f"Error calculating collection times for post created at {post_created_at}: {e}")
        return []


def should_collect_metrics(post_id: str, platform: str, timeslot_hours: int) -> bool:
    """
    Check if metrics have already been collected for this post at this timeslot

    Args:
        post_id: Post identifier
        platform: Platform name
        timeslot_hours: Hours since post creation

    Returns:
        True if metrics should be collected, False if already collected
    """
    try:
        # Check if snapshot already exists for this timeslot
        res = db.supabase.table("post_snapshots") \
            .select("id") \
            .eq("post_id", post_id) \
            .eq("platform", platform) \
            .eq("timeslot_hours", timeslot_hours) \
            .execute()

        # If no existing snapshot, we should collect
        return len(res.data or []) == 0

    except Exception as e:
        logger.error(f"Error checking if metrics already collected for {post_id}: {e}")
        # On error, assume we should collect to be safe
        return True


async def collect_and_store_metrics(post: Dict[str, Any]) -> bool:
    """
    Collect metrics for a post and store them in the database

    Args:
        post: Post data dict with post_id, platform, business_id, media_id, created_at

    Returns:
        True if metrics were collected and stored, False otherwise
    """
    post_id = post["post_id"]
    platform = post["platform"]
    business_id = post["business_id"]
    media_id = post["media_id"]

    logger.info(f"🔍 Checking metrics collection for post {post_id} on {platform}")

    # Calculate collection times
    collection_times = calculate_collection_times(post["created_at"])

    # Check which intervals are due for collection
    intervals_to_collect = []
    for collection_time in collection_times:
        if collection_time["is_due"]:
            if should_collect_metrics(post_id, platform, collection_time["hours"]):
                intervals_to_collect.append(collection_time["hours"])

    if not intervals_to_collect:
        logger.debug(f"⏳ No metrics collection due for post {post_id}")
        return False

    # Get platform credentials
    credentials = get_platform_credentials(platform, business_id)
    if not credentials:
        logger.error(f"❌ No credentials found for {platform} business {business_id}")
        return False

    # Collect metrics for each due interval
    metrics_collected = False

    for timeslot_hours in intervals_to_collect:
        try:
            logger.info(f"📊 Collecting {timeslot_hours}h metrics for post {post_id}")

            # Fetch metrics from platform API
            if platform == "facebook":
                metrics = await fetch_facebook_post_metrics(
                    credentials["page_id"],
                    credentials["access_token"],
                    media_id
                )
            elif platform == "instagram":
                metrics = await fetch_instagram_post_metrics(
                    credentials["page_id"],  # This should be Instagram account ID
                    credentials["access_token"],
                    media_id
                )
            else:
                logger.warning(f"⚠️ Metrics collection not implemented for {platform}")
                continue

            if "error" in metrics:
                logger.error(f"❌ Failed to fetch metrics for {post_id}: {metrics['error']}")
                continue

            # Store metrics in database
            db.insert_post_snapshot(
                post_id=post_id,
                platform=platform,
                metrics=metrics,
                profile_id=business_id,
                timeslot_hours=timeslot_hours
            )

            logger.info(f"✅ Stored {timeslot_hours}h metrics for post {post_id}: {metrics}")
            metrics_collected = True

        except Exception as e:
            logger.error(f"❌ Error collecting {timeslot_hours}h metrics for post {post_id}: {e}")
            continue

    return metrics_collected


async def run_metrics_collection_job():
    """
    Main job function to collect metrics for all eligible posts
    """
    logger.info("🚀 Starting social media metrics collection job")

    try:
        # Get recently posted content
        posts = get_recently_posted_content(hours_threshold=200)  # Look back 200 hours to catch all intervals

        if not posts:
            logger.info("ℹ️ No posts found for metrics collection")
            return

        total_processed = 0
        metrics_collected = 0

        # Process each post
        for post in posts:
            try:
                if await collect_and_store_metrics(post):
                    metrics_collected += 1
                total_processed += 1

            except Exception as e:
                logger.error(f"❌ Error processing post {post['post_id']}: {e}")
                continue

        logger.info(f"✅ Metrics collection job completed: {metrics_collected}/{total_processed} posts had new metrics collected")

    except Exception as e:
        logger.error(f"❌ Critical error in metrics collection job: {e}")


async def run_continuous_metrics_collection():
    """
    Run metrics collection continuously with a check interval
    """
    logger.info("🔄 Starting continuous metrics collection (checks every 30 minutes)")

    while True:
        try:
            await run_metrics_collection_job()

            # Wait 30 minutes before next check
            logger.info("⏰ Waiting 30 minutes before next metrics collection check...")
            await asyncio.sleep(30 * 60)  # 30 minutes

        except KeyboardInterrupt:
            logger.info("🛑 Metrics collection stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Error in continuous collection loop: {e}")
            # Wait a bit before retrying
            await asyncio.sleep(5 * 60)  # 5 minutes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Social Media Metrics Collector")
    parser.add_argument(
        "mode",
        choices=["once", "continuous"],
        help="Run once or continuously"
    )

    args = parser.parse_args()

    if args.mode == "once":
        # Run once and exit
        asyncio.run(run_metrics_collection_job())
    elif args.mode == "continuous":
        # Run continuously
        asyncio.run(run_continuous_metrics_collection())