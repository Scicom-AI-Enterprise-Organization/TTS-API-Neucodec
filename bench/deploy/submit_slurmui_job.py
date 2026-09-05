#!/usr/bin/env python3
"""
Submit (or replace) a SlurmUI job from a GitOps manifest through the public API -- without
the GitOps reconciler. Reads `metadata.cluster`, `metadata.name`, `spec.partition` and
`spec.script` from the yaml, cancels any PENDING/RUNNING job of that name on the cluster (the
prod script binds a fixed GPU and port, so two copies cannot coexist), submits the script with
`POST /api/v1/clusters/:cluster/jobs`, then follows the job until its output says SERVING.

Auth: SLURMUI_URL + SLURMUI_API_KEY (an `aura_…` Bearer token) from the environment.

    set -a; source .env; set +a
    uv run --with pyyaml python bench/deploy/submit_slurmui_job.py \
        /Users/husein.z/Documents/ucc_slurm-ui-job/jobs/tm-h20/tts-api.yaml

    --dry-run        show what would be cancelled/submitted, do nothing
    --no-cancel      leave existing jobs alone (they will collide on GPU/port -- only for a
                     script that was edited to use another GPU/port)
    --drop-account   strip `#SBATCH --account=…` (needed when the token owner's Linux user is
                     not in that Slurm account)
    --wait MIN       how long to follow the new job (default 25 min; a cold bootstrap of the
                     venv on NFS takes longer than a warm start's ~3 min)
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit('needs PyYAML: uv run --with pyyaml python ' + ' '.join(sys.argv))


def api(method, path, body=None):
    url = os.environ['SLURMUI_URL'].rstrip('/') + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        'Authorization': f"Bearer {os.environ['SLURMUI_API_KEY']}",
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors='replace')
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {'error': raw[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('yaml_path')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-cancel', action='store_true')
    ap.add_argument('--drop-account', action='store_true')
    ap.add_argument('--wait', type=float, default=25.0, help='minutes to follow the job')
    args = ap.parse_args()
    for k in ('SLURMUI_URL', 'SLURMUI_API_KEY'):
        if not os.environ.get(k):
            sys.exit(f'{k} not set (set -a; source .env; set +a)')

    m = yaml.safe_load(open(args.yaml_path))
    cluster = m['metadata']['cluster']
    name = m['metadata']['name']
    partition = m['spec'].get('partition')
    script = m['spec']['script']
    if args.drop_account:
        script = re.sub(r'^#SBATCH\s+--account=\S+\n', '', script, flags=re.M)
    if not re.search(r'^#SBATCH\s+--job-name=', script, flags=re.M):
        script = script.replace('#!/bin/bash\n', f'#!/bin/bash\n#SBATCH --job-name={name}\n', 1)
    print(f'manifest: name={name} cluster={cluster} partition={partition} script={len(script)} chars')

    # 0. does the token work at all?
    code, clusters = api('GET', '/api/v1/clusters')
    if code != 200:
        sys.exit(f'GET /api/v1/clusters -> {code} {clusters}: the SLURMUI_API_KEY is not accepted '
                 f'(not an aura_ token of this deployment, or revoked). Mint one at /profile/api-tokens.')
    names = [c.get('name') for c in (clusters if isinstance(clusters, list) else clusters.get('clusters', []))]
    print(f'token ok; clusters visible: {names}')

    # 1. existing jobs of that name
    code, listing = api('GET', f'/api/v1/clusters/{cluster}/jobs?name={name}&limit=20')
    if code != 200:
        sys.exit(f'list jobs -> {code} {listing}')
    rows = listing.get('jobs') or listing.get('data') or listing.get('items') or []
    # the list is already filtered by name (substring); rows come back with name=None, so
    # only exclude rows whose name is set and clearly something else
    live = [j for j in rows if str(j.get('status', '')).upper() in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING')
            and (not j.get('name') or name in str(j.get('name')))]
    print(f'{len(rows)} job rows named like {name!r}; live: '
          + ', '.join(f"{j.get('id')} (slurm {j.get('slurmJobId')}, {j.get('status')})" for j in live))

    if args.dry_run:
        print('dry run: would cancel the live rows above and submit the script. Nothing done.')
        return

    # 2. cancel live ones (same GPU/port as the new job)
    if live and not args.no_cancel:
        for j in live:
            code, out = api('POST', f"/api/v1/jobs/{j['id']}/cancel")
            print(f"cancel {j['id']} (slurm {j.get('slurmJobId')}) -> {code} {out}")
        time.sleep(8)      # let slurm reap the process group before the new job binds the port

    # 3. submit
    body = {'script': script, 'name': name}
    if partition:
        body['partition'] = partition
    code, job = api('POST', f'/api/v1/clusters/{cluster}/jobs', body)
    if code not in (200, 201):
        sys.exit(f'submit -> {code} {job}')
    job_id = job.get('id')
    print(f"submitted: id={job_id} slurmJobId={job.get('slurmJobId')} status={job.get('status')}")

    # 4. follow until SERVING / failure / timeout
    deadline = time.time() + args.wait * 60
    seen = ''
    while time.time() < deadline:
        time.sleep(20)
        code, detail = api('GET', f'/api/v1/jobs/{job_id}?output=1')
        if code != 200:
            print(f'poll -> {code} {detail}')
            continue
        status = str(detail.get('status', '')).upper()
        out = detail.get('output') or detail.get('stdout') or ''
        new = out[len(seen):] if out.startswith(seen) else out
        seen = out
        for line in new.splitlines():
            if line.startswith(('>>', '!!')) or 'Uvicorn running' in line or 'SERVING' in line or 'Error' in line:
                print(f'  {line[:200]}')
        print(f'  status={status} slurm={detail.get("slurmJobId")}')
        if 'SERVING' in out:
            print('job is serving')
            return
        if status in ('FAILED', 'CANCELLED', 'COMPLETED', 'TIMEOUT', 'NODE_FAIL'):
            print(f'job ended: {status}. Last output lines:')
            print('\n'.join(out.splitlines()[-15:]))
            sys.exit(1)
    print('still starting when the wait ran out; check the job page / logs')


if __name__ == '__main__':
    main()
