# Test & Validation Prompt — POST /generate

## CONTEXT BLOCK (paste before this prompt)

```
PROJECT CONTEXT:
- Backend: Python + FastAPI campaign AI engine
- Pipeline: 8 stages (1, 2, 3, 4, 5, 6, 7b, 7, 8) chained via _run_stage()
- Checkpoint files saved at: checkpoints/{job_id}/stage_{N}.json
- utils/llm_runtime.py exports use_mock_llm() which reads USE_MOCK_LLM env var
- utils/claude_client.py exports call_claude()
- Test payload file: test_payload.json (in project root)
- DO NOT modify any pipeline logic. This prompt is for testing and validation only.
```

---

## STEP 1 — Run mock mode test (zero API cost)

Do the following in order. Stop and report any error before continuing.

```bash
# 1. Set mock mode so no Claude API calls are made
export USE_MOCK_LLM=true
export ANTHROPIC_API_KEY=sk-placeholder

# 2. Start the server in the background
uvicorn main:app --port 8000 &
sleep 3

# 3. Send the test payload to /generate
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d @test_payload.json \
  | python3 -m json.tool > mock_response.json

echo "Exit code: $?"
echo "Response saved to mock_response.json"
```

After running, confirm:
- Exit code is 0
- mock_response.json exists and is valid JSON
- It contains the keys: strategy, calendar, influencer_matches, influencer_strategy_note, influencer_stage_skipped
- influencer_stage_skipped is false (because influencer_candidates was provided)
- influencer_matches is a list with at least 1 item

Print a summary of what was found in each key. If any key is missing, show the actual
response and identify which stage likely failed by checking the checkpoint files.

---

## STEP 2 — Validate checkpoint files

```bash
# List all checkpoint files created for job test-job-001
find checkpoints/test-job-001 -name "*.json" | sort

# Print the keys of each checkpoint (not full content — just top-level keys)
for f in checkpoints/test-job-001/*.json; do
  echo "--- $f ---"
  python3 -c "import json,sys; d=json.load(open('$f')); print(list(d.keys()))"
done
```

Expected checkpoint files:
- stage_1.json  — raw brief dict (all CampaignBrief fields)
- stage_2.json  — must contain: business_summary, tone_descriptor, tone_guidelines, budget_tier
- stage_3.json  — must contain: content_gaps, positioning_opportunity, channels_to_prioritize
- stage_4.json  — must contain: persona_name, pain_points, messaging_hooks, platform_behaviour
- stage_5.json  — must contain: campaign_summary, core_message, content_pillars, kpis, budget_allocation
- stage_6.json  — must contain: platform_content, campaign_hashtag_set, posting_frequency
- stage_7b.json — must contain: influencer_matches, influencer_strategy_note
- stage_7.json  — must contain: ml_score, ml_verdict, predicted_roi, written_explanation
- stage_8.json  — must contain: days, start_date, total_days

Report which checkpoints exist and which keys are missing from any of them.
If a checkpoint is missing entirely, report which stage failed.

---

## STEP 3 — Validate stage 7b influencer output specifically

```bash
python3 -c "
import json

with open('checkpoints/test-job-001/stage_7b.json') as f:
    data = json.load(f)

print('influencer_stage_skipped:', data.get('influencer_stage_skipped'))
matches = data.get('influencer_matches', [])
print(f'Number of matches: {len(matches)}')

for i, m in enumerate(matches):
    print(f'\nMatch {i+1}:')
    print(f'  influencer_id:              {m.get(\"influencer_id\")}')
    print(f'  fit_score:                  {m.get(\"fit_score\")}')
    print(f'  suggested_collaboration:    {m.get(\"suggested_collaboration_type\")}')
    print(f'  suggested_budget_usd:       {m.get(\"suggested_budget_usd\")}')
    print(f'  outreach_message length:    {len(m.get(\"outreach_message\", \"\"))} chars')
    print(f'  fit_reasoning:              {m.get(\"fit_reasoning\", \"\")[:80]}...')

print('\nstrategy_note:', data.get('influencer_strategy_note', '')[:120], '...')
"
```

Expected results:
- influencer_stage_skipped: False
- Between 1 and 3 matches (mock returns 2)
- Each match has: influencer_id (int), fit_score (float), suggested_collaboration_type (str),
  suggested_budget_usd (float), outreach_message (str), fit_reasoning (str)
- Budget splits should sum to approximately 20% of 5000 USD = 1000 USD total

Report any missing fields or unexpected values.

---

## STEP 4 — Test resume from checkpoint (pipeline resilience)

```bash
# Delete only stage 6 and later checkpoints to simulate a mid-pipeline failure
rm checkpoints/test-job-001/stage_6.json
rm -f checkpoints/test-job-001/stage_7b.json
rm -f checkpoints/test-job-001/stage_7.json
rm -f checkpoints/test-job-001/stage_8.json

echo "Deleted stages 6–8 checkpoints. Stages 1–5 still cached."

# Re-run the pipeline with the same job_id — should resume from stage 6
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d @test_payload.json \
  | python3 -m json.tool > resume_response.json

echo "Resume test complete. Check resume_response.json"
```

Confirm:
- The pipeline completed without errors
- Stages 1–5 were loaded from cache (no new mock LLM calls for those stages)
- Stages 6–8 and 7b were re-run
- resume_response.json has the same top-level keys as mock_response.json

---

## STEP 5 — Test stage 7b skip behaviour (no influencer candidates)

```bash
# Create a stripped payload without influencer_candidates
python3 -c "
import json
with open('test_payload.json') as f:
    payload = json.load(f)

payload['job_id'] = 'test-job-no-influencers'
payload['influencer_candidates'] = []

with open('test_payload_no_influencers.json', 'w') as f:
    json.dump(payload, f, indent=2)
print('Created test_payload_no_influencers.json')
"

curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d @test_payload_no_influencers.json \
  | python3 -c "
import json, sys
r = json.load(sys.stdin)
print('influencer_stage_skipped:', r.get('influencer_stage_skipped'))
print('influencer_matches count:', len(r.get('influencer_matches', [])))
print('Pipeline still completed:', 'calendar' in r and 'strategy' in r)
"
```

Expected:
- influencer_stage_skipped: True
- influencer_matches: [] (empty list)
- Pipeline still completed successfully (calendar and strategy present)

---

## STEP 6 — Test the /generate/stream SSE endpoint

```bash
# Test SSE streaming — should receive stage_complete events then a final complete event
curl -s -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d @test_payload.json \
  --no-buffer 2>&1 | head -60
```

Expected output format (one line per event):
```
data: {"event": "stage_complete", "stage": 1, "stage_name": "Data Collection", "progress": 8}

data: {"event": "stage_complete", "stage": "2", "stage_name": "Business Analysis", "progress": 18}

... (one per stage) ...

data: {"event": "complete", "progress": 100, "result": { ... full response ... }}
```

Confirm:
- Events arrive in order (stage 1 → 2 → 3 → 4 → 5 → 6 → 7b → 7 → 8 → complete)
- Progress numbers are strictly increasing
- The final "complete" event contains a "result" key with strategy, calendar, influencer_matches

If any event is missing or out of order, check stream_pipeline() in main.py for the stage ordering.

---

## STEP 7 — Real API smoke test (uses Claude credits — run last)

Only run this after all mock tests pass.

```bash
# Switch off mock mode
unset USE_MOCK_LLM
export ANTHROPIC_API_KEY=your_real_key_here

# Use a fresh job_id so no checkpoints interfere
python3 -c "
import json
with open('test_payload.json') as f:
    p = json.load(f)
p['job_id'] = 'real-test-001'
print(json.dumps(p))
" | curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d @- \
  | python3 -c "
import json, sys
r = json.load(sys.stdin)
s = r.get('strategy', {})
print('=== REAL API RESULT ===')
print('Campaign name:      ', s.get('campaign_summary', {}).get('name', 'MISSING'))
print('Core message:       ', s.get('core_message', 'MISSING')[:80])
print('Tone descriptor:    ', s.get('campaign_summary', {}).get('tagline', 'MISSING')[:60])
matches = r.get('influencer_matches', [])
print(f'Influencer matches: {len(matches)}')
for m in matches:
    print(f'  ID {m[\"influencer_id\"]} — score {m[\"fit_score\"]} — {m[\"suggested_collaboration_type\"]}')
cal = r.get('calendar', {})
print('Calendar days:      ', cal.get('total_days', 'MISSING'))
print('ML verdict:         ', r.get('strategy', {}).get('ml_verdict', 'check stage7 checkpoint'))
"
```

Expected real API results:
- Campaign name is creative and relevant to BrewMate
- Core message references coffee, remote work, or the AI brewing angle
- Influencer matches: 2–3 results — influencer IDs 1, 2, or 3 most likely (best platform/niche fit)
  - ID 4 (YouTube, tech, 3.1% engagement) may score below 5.0 threshold — acceptable
  - ID 5 (food blogger, 2.4% engagement) should score lowest — may be excluded
- Calendar total_days should be 42 (6 weeks × 7)
- Cost for this single run: approximately $0.04–0.06 USD

---

## What to do if something fails

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `KeyError: 'platform_content'` in stage 8 | Stage 6 mock still returns old shape | Re-apply Prompt 8 to stage6_tactical.py |
| `influencer_stage_skipped: True` despite sending candidates | `influencer_candidates` key not parsed by Pydantic | Check InfluencerProfile model in schemas.py is defined and added to CampaignBrief |
| Stage "7b" checkpoint not found | `int \| str` type hint not updated in CheckpointManager | Re-apply Prompt 10 Edit 2 to pipeline.py |
| SSE stream returns only one event then stops | Exception in stream_pipeline generator | Check main.py stream_pipeline try/except — error event should show the stage name |
| Real API returns invalid JSON error | Prompt too long for Haiku context | Check stage 6 prompt — if influencer_candidates list is very large it may exceed limits |
| `RuntimeError: ML_ROOT_PATH not set` in stage 7 | Running real mode without ML model | Set `USE_MOCK_LLM=true` OR set ML_ROOT_PATH to your ML project path |
