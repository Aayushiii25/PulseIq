"""
backend/scheduler.py — APScheduler for continuous ingestion
============================================================
Runs the pipeline periodically in the background.
"""
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from backend.pipeline import run_pipeline
from config import settings

log = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

def scheduled_job():
    log.info("⏰ Running scheduled PulseIQ pipeline...")
    try:
        run_pipeline(skip_fetch=False, api_key=settings.news_api_key)
        log.info("✅ Scheduled pipeline run complete.")
    except Exception as e:
        log.error(f"❌ Scheduled pipeline run failed: {e}")

def start_scheduler():
    if not scheduler.running:
        minutes = settings.scheduler_interval_minutes
        scheduler.add_job(
            scheduled_job,
            trigger=IntervalTrigger(minutes=minutes),
            id="pulseiq_pipeline",
            name="Run ML Pipeline periodically",
            replace_existing=True,
        )
        scheduler.start()
        log.info(f"🚀 Scheduler started. Running every {minutes} minutes.")

def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        log.info("🛑 Scheduler stopped.")
