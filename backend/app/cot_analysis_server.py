#!/usr/bin/env python3
"""
Standalone CoT Analysis Server

This server runs in isolation to perform a single CoT analysis job.
It communicates with the main server via HTTP and shared state (job_db.json).
"""

import asyncio
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.runner import job_db, save_job_db, get_job_raw_data
from app.path_manager import path_manager
from app.cot_eval_v2.evaluator import PillarsEvaluator
from app.cot_eval_v2.judge import Judge

app = FastAPI(title="CoT Analysis Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging - configure to write to both stdout (for subprocess capture) and a dedicated log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Goes to subprocess stdout (captured in log file)
        logging.StreamHandler(sys.stderr)    # Also log to stderr
    ]
)
logger = logging.getLogger(__name__)

# Track running analyses to prevent duplicates (module-level set)
_running_analyses = set()

def run_cot_analysis_sync(cot_job_id: str, job_id: str, config_dict: Dict[str, Any]):
    """
    Synchronous function to run CoT analysis.
    
    Args:
        cot_job_id: The CoT analysis job ID
        job_id: The parent job ID
        config_dict: Configuration dict with judge_mode, diagnostic, etc.
    """
    start_time = time.time()
    judge_mode = config_dict.get("judge_mode", "ALWAYS")
    diagnostic = config_dict.get("diagnostic", False)
    
    try:
        # Get raw data
        raw_data = get_job_raw_data(job_id)
        data_list = raw_data.get('data', [])
        
        if not data_list:
            error_msg = "No data found for analysis"
            logger.error(f"Error - {error_msg} - cot_job_id={cot_job_id}, job_id={job_id}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        total_samples = len(data_list)
        logger.info(f"Analysis initialized - cot_job_id={cot_job_id}, total_samples={total_samples}")
        
        # Update progress: starting
        queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
        if queue_entry:
            queue_entry["status"] = "RUNNING"
            queue_entry["started_at"] = time.time()
            queue_entry["progress"]["total_samples"] = total_samples
            queue_entry["progress"]["start_time"] = time.time()
            queue_entry["progress"]["current_activity"] = "Initializing analysis..."
            save_job_db()
        
        # Initialize evaluator and judge
        # Get configuration from path_manager
        path_config = path_manager.get_config()
        
        # Create judge if OpenAI key available
        judge = None
        if path_config.openai_api_key:
            # Set the API key as environment variable for OpenAI client
            import os
            os.environ['OPENAI_API_KEY'] = path_config.openai_api_key
            judge = Judge(mode=judge_mode, diagnostic=diagnostic)
        
        # Initialize evaluator
        evaluator = PillarsEvaluator(judge=judge)
        
        logger.info(f"Starting CoT analysis for {total_samples} samples...")
        print(f"🔍 Starting CoT analysis for {total_samples} samples...", flush=True)
        
        results = []
        for i, sample in enumerate(data_list):
            # Update progress
            queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
            if queue_entry:
                queue_entry["progress"]["current_sample"] = i
                queue_entry["progress"]["processed_samples"] = i + 1
                queue_entry["progress"]["percentage"] = ((i + 1) / total_samples) * 100.0
                queue_entry["progress"]["current_activity"] = f"Analyzing sample {i+1}/{total_samples}..."
                
                # Calculate estimated time remaining
                if i > 0:
                    elapsed = time.time() - queue_entry["progress"]["start_time"]
                    avg_time_per_sample = elapsed / (i + 1)
                    remaining_samples = total_samples - (i + 1)
                    queue_entry["progress"]["estimated_time_remaining"] = avg_time_per_sample * remaining_samples
                
                # Save progress every 10 samples
                if (i + 1) % 10 == 0 or i == 0 or i == total_samples - 1:
                    save_job_db()
            
            try:
                # Extract model reasoning and answer
                model_output = sample.get("output", "")
                question = sample.get("question", "")
                gt = sample.get("gt", "")
                
                # Parse model output to extract reasoning and answer
                if "####" in model_output:
                    parts = model_output.split("####")
                    model_reasoning_text = parts[0].strip()
                    predicted_answer_text = parts[1].strip() if len(parts) > 1 else ""
                else:
                    model_reasoning_text = model_output.strip()
                    predicted_answer_text = ""
                
                # Construct full CoT text
                if predicted_answer_text:
                    full_cot_text = f"{model_reasoning_text}\n#### {predicted_answer_text}"
                else:
                    full_cot_text = model_reasoning_text
                
                # Run analysis using the analyze method
                flags, evidence, rule_scores_dict, judge_scores_dict, fused_scores = evaluator.analyze(
                    problem=question,
                    cot_text=full_cot_text,
                    gold=gt
                )
                
                # Get existing correctness from sample data if available
                is_correct = sample.get('score', [False])[0] if isinstance(sample.get('score'), list) else sample.get('score', False)
                if is_correct is not None:
                    evidence['final_correct'] = bool(is_correct)
                    # Recalculate rule scores with correct final_correct value
                    from app.cot_eval_v2.scoring import rule_scores
                    rule_scores_dict = rule_scores(evidence)
                    # Recalculate fused scores
                    from app.cot_eval_v2.scoring import fuse_with_judge
                    fused_scores = fuse_with_judge(rule_scores_dict, judge_scores_dict, evidence)
                
                # Convert flags to dictionary format
                flags_dict = {}
                for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
                    pillar_flags = flags.get_flags_by_pillar(pillar)
                    flags_dict[pillar] = [flag.to_dict() for flag in pillar_flags]
                
                # Build analysis result in the expected format
                analysis_result = {
                    "faithfulness": fused_scores.get("faithfulness", 0.0),
                    "utility": fused_scores.get("utility", 0.0),
                    "coherence": fused_scores.get("coherence", 0.0),
                    "factuality": fused_scores.get("factuality", 0.0),
                    "overall": (fused_scores.get("faithfulness", 0.0) + 
                               fused_scores.get("utility", 0.0) + 
                               fused_scores.get("coherence", 0.0) + 
                               fused_scores.get("factuality", 0.0)) / 4.0,
                    "flags": flags_dict,
                    "evidence": evidence,
                    "rule_scores": rule_scores_dict,
                    "judge_scores": judge_scores_dict
                }
                
                logger.info(f"Sample {i+1}/{total_samples} analyzed")
                if (i + 1) % 10 == 0 or i == 0 or i == total_samples - 1:
                    print(f"   ✅ Processed {i+1}/{total_samples} samples ({((i+1)/total_samples*100):.1f}%)", flush=True)
                
                # Store result
                results.append({
                    "idx": sample.get("idx", i),
                    "question": question,
                    "ground_truth": gt,
                    "model_output": model_output,
                    "cot_text": full_cot_text,
                    "analysis": analysis_result,
                    "faithfulness": analysis_result.get("faithfulness", 0.0),
                    "utility": analysis_result.get("utility", 0.0),
                    "coherence": analysis_result.get("coherence", 0.0),
                    "factuality": analysis_result.get("factuality", 0.0),
                    "overall": analysis_result.get("overall", 0.0),
                    "flags": analysis_result.get("flags", {}),
                    "judge_scores": judge_scores_dict
                })
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                results.append({
                    "idx": sample.get("idx", i),
                    "question": sample.get("question", ""),
                    "ground_truth": sample.get("gt", ""),
                    "model_output": sample.get("output", ""),
                    "cot_text": "",
                    "analysis": {},
                    "faithfulness": 0.0,
                    "utility": 0.0,
                    "coherence": 0.0,
                    "factuality": 0.0,
                    "overall": 0.0,
                    "flags": [],
                    "error": str(e)
                })
        
        # Calculate summary statistics
        total_samples = len(results)
        if total_samples > 0:
            avg_faithfulness = sum(r.get("faithfulness", 0.0) for r in results) / total_samples
            avg_utility = sum(r.get("utility", 0.0) for r in results) / total_samples
            avg_coherence = sum(r.get("coherence", 0.0) for r in results) / total_samples
            avg_factuality = sum(r.get("factuality", 0.0) for r in results) / total_samples
            avg_overall = sum(r.get("overall", 0.0) for r in results) / total_samples
            
            # Count flags (flags is now a dict with pillar keys)
            total_flags = sum(sum(len(flags_list) for flags_list in r.get("flags", {}).values()) for r in results)
            flags_by_pillar = {
                "faithfulness": sum(len(r.get("flags", {}).get("faithfulness", [])) for r in results),
                "utility": sum(len(r.get("flags", {}).get("utility", [])) for r in results),
                "coherence": sum(len(r.get("flags", {}).get("coherence", [])) for r in results),
                "factuality": sum(len(r.get("flags", {}).get("factuality", [])) for r in results)
            }
            
            # Judge statistics
            judge_calls = sum(1 for r in results if r.get("judge_scores") is not None)
            judge_call_rate = judge_calls / total_samples if total_samples > 0 else 0.0
        else:
            avg_faithfulness = avg_utility = avg_coherence = avg_factuality = avg_overall = 0.0
            total_flags = 0
            flags_by_pillar = {}
            judge_call_rate = 0.0
        
        analysis_time = time.time() - start_time
        
        # Build analysis result
        analysis_result = {
            "job_id": job_id,
            "cot_job_id": cot_job_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_method": "pillars_v2",
            "config": {
                "judge_mode": judge_mode,
                "diagnostic": diagnostic
            },
            "summary": {
                "total_samples": total_samples,
                "avg_faithfulness": avg_faithfulness,
                "avg_utility": avg_utility,
                "avg_coherence": avg_coherence,
                "avg_factuality": avg_factuality,
                "avg_overall": avg_overall,
                "total_flags": total_flags,
                "flags_by_pillar": flags_by_pillar,
                "judge_call_rate": judge_call_rate,
                "judge_budget_used": judge_calls if judge else 0,
                "judge_budget_total": total_samples if judge else 0,
                "analysis_time": analysis_time,
                "avg_time_per_sample": analysis_time / total_samples if total_samples > 0 else 0.0
            },
            "per_sample": results
        }
        
        logger.info(f"CoT analysis complete! Processed {len(results)} samples")
        print(f"🎉 CoT analysis complete! Processed {len(results)} samples", flush=True)
        
        # Save results
        queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
        if queue_entry:
            parent_job = job_db.get(job_id, {})
            model_name = parent_job.get("request", {}).get("model", "unknown")
            dataset = parent_job.get("request", {}).get("dataset", "unknown")
            
            # Create organized folder structure
            exports_dir = Path(path_manager.get_config().exports_dir)
            timestamp_str = analysis_result.get('timestamp', time.strftime("%Y%m%d_%H%M%S")).replace(' ', '_').replace(':', '')
            judge_mode_str = config_dict.get("judge_mode", "ALWAYS")
            organized_dir = exports_dir / "cot_analysis" / model_name / dataset / f"{job_id}_{timestamp_str}_{judge_mode_str}"
            organized_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            cot_analysis_file = organized_dir / f"cot_analysis_{job_id}.json"
            with open(cot_analysis_file, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            queue_entry["json_path"] = str(cot_analysis_file)
            queue_entry["status"] = "DONE"
            queue_entry["completed_at"] = time.time()
            queue_entry["progress"]["percentage"] = 100.0
            queue_entry["progress"]["current_activity"] = "Analysis complete!"
            save_job_db()
            
            logger.info(f"CoT analysis saved to: {cot_analysis_file}")
            
            # Schedule server shutdown after a brief delay to allow response
            logger.info("Analysis complete, server will shutdown shortly...")
            import threading
            def shutdown_delayed():
                time.sleep(2)  # Give time for any final operations
                import signal
                import os
                os.kill(os.getpid(), signal.SIGTERM)
            threading.Thread(target=shutdown_delayed, daemon=True).start()
        
    except Exception as e:
        logger.error(f"Error in CoT analysis: {e}", exc_info=True)
        queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
        if queue_entry:
            queue_entry["status"] = "ERROR"
            queue_entry["error"] = str(e)
            queue_entry["completed_at"] = time.time()
            save_job_db()
        raise
    finally:
        # Clean up running set
        _running_analyses.discard(cot_job_id)

@app.post("/run-analysis")
async def run_analysis(request: Dict[str, Any]):
    """Endpoint to trigger the analysis"""
    cot_job_id = request.get("cot_job_id")
    job_id = request.get("job_id")
    config = request.get("config", {})
    
    if not cot_job_id or not job_id:
        raise HTTPException(status_code=400, detail="Missing cot_job_id or job_id")
    
    # Check if analysis is already running
    if cot_job_id in _running_analyses:
        logger.warning(f"Analysis already running for cot_job_id={cot_job_id}, ignoring duplicate request")
        queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
        if queue_entry and queue_entry.get("status") == "RUNNING":
            return {"status": "already_running", "cot_job_id": cot_job_id, "message": "Analysis already in progress"}
    
    # Also check job_db status
    queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
    if queue_entry and queue_entry.get("status") == "RUNNING":
        logger.warning(f"Analysis already marked as RUNNING in job_db for cot_job_id={cot_job_id}, ignoring duplicate request")
        return {"status": "already_running", "cot_job_id": cot_job_id, "message": "Analysis already in progress"}
    
    try:
        # Mark as running
        _running_analyses.add(cot_job_id)
        
        # Run in executor to avoid blocking the server
        loop = asyncio.get_event_loop()
        # Start the analysis in background - don't wait for completion
        loop.run_in_executor(None, run_cot_analysis_sync, cot_job_id, job_id, config)
        
        logger.info(f"Analysis started for cot_job_id={cot_job_id}")
        return {"status": "started", "cot_job_id": cot_job_id, "message": "Analysis started in background"}
    except Exception as e:
        # Remove from running set on error
        _running_analyses.discard(cot_job_id)
        logger.error(f"Error starting analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone CoT Analysis Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    logger.info(f"Starting CoT Analysis Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

