"""GENESIS V14 - Ultra-Meta Cognitive OS"""
import os, time, random, hashlib
from datetime import datetime
import numpy as np
import psutil
from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse
import uvicorn

class HyperdimensionalVector:
    DIM = 4096
    @staticmethod
    def random():
        return np.random.choice([-1, 1], size=4096)
    @staticmethod
    def bind(a, b):
        return a * b
    @staticmethod
    def similarity(a, b):
        return float(np.dot(a, b)) / 4096

class SpikingNeuron:
    def __init__(self, threshold=1.0):
        self.potential = 0.0
        self.threshold = threshold
        self.spikes = 0
    def input(self, signal):
        self.potential += signal
        if self.potential >= self.threshold:
            self.potential = 0.0
            self.spikes += 1
            return True
        return False

class VSCEngine:
    @staticmethod
    def convolve(a, b):
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))

class FreeEnergyMinimizer:
    def __init__(self):
        self.beliefs = np.random.randn(64)
    def update(self, obs):
        err = obs - self.beliefs
        self.beliefs += 0.1 * err
        return float(np.sum(err ** 2))

class Agent:
    def __init__(self, name, role="worker"):
        self.name = name
        self.role = role
        self.id = hashlib.sha256(f"{name}{time.time()}".encode()).hexdigest()[:12]
        self.created = datetime.utcnow().isoformat()
        self.status = "online"
        self.ops_count = 0
        self.neuron = SpikingNeuron()
        self.hd_vector = HyperdimensionalVector.random()
    def process(self, signal=None):
        if signal is None:
            signal = random.random()
        spiked = self.neuron.input(signal)
        self.ops_count += 1
        return {"agent": self.name, "spiked": spiked, "ops": self.ops_count}
    def to_dict(self):
        return {"id": self.id, "name": self.name, "role": self.role, "status": self.status, "created": self.created, "ops_count": self.ops_count, "spikes": self.neuron.spikes}

class DigitalTwin:
    def __init__(self, agent):
        self.agent_id = agent.id
        self.agent_name = agent.name
        self.snapshot_time = datetime.utcnow().isoformat()
        self.ops_snapshot = agent.ops_count
        self.spikes_snapshot = agent.neuron.spikes
        self.potential_snapshot = agent.neuron.potential
        self.beliefs = FreeEnergyMinimizer()
    def predict_next(self, steps=5):
        predictions = []
        neuron = SpikingNeuron()
        neuron.potential = self.potential_snapshot
        for i in range(steps):
            signal = random.random()
            spiked = neuron.input(signal)
            fe = self.beliefs.update(np.random.randn(64))
            predictions.append({"step": i+1, "spiked": spiked, "potential": round(neuron.potential, 4), "free_energy": round(fe, 4)})
        return predictions
    def to_dict(self):
        return {"agent_id": self.agent_id, "agent_name": self.agent_name, "snapshot_time": self.snapshot_time, "ops_snapshot": self.ops_snapshot, "spikes_snapshot": self.spikes_snapshot}

BOOT_TIME = datetime.utcnow().isoformat()
AGENTS = [Agent("Orchestrator", "master"), Agent("SNN-Core", "compute"), Agent("HDC-Memory", "memory"), Agent("VSC-Transform", "transform"), Agent("FreeEnergy-Optimizer", "optimizer"), Agent("TwinManager", "twin")]
FREE_ENERGY = FreeEnergyMinimizer()
DEPLOY_TARGETS = {"replit": {"status": "ready", "url": ""}, "vercel": {"status": "ready", "url": ""}, "netlify": {"status": "ready", "url": ""}, "oracle": {"status": "ready", "url": ""}, "ghcr": {"status": "ready", "url": "ghcr.io/adaojoaquim/genesis-v14"}, "huggingface": {"status": "ready", "url": ""}}

health_router = APIRouter(tags=["health"])
@health_router.get("/health")
def health_check():
    return {"status": "healthy", "version": "14.0.0", "uptime_since": BOOT_TIME, "cpu_percent": psutil.cpu_percent(), "memory_mb": round(psutil.virtual_memory().used / 1e6, 1)}

agent_router = APIRouter(tags=["agents"])
@agent_router.get("/agents")
def get_agents():
    return {"agents": [a.to_dict() for a in AGENTS], "count": len(AGENTS)}
@agent_router.get("/ops")
def get_ops():
    results = [a.process(random.random()) for a in AGENTS]
    return {"ops_batch": results, "total_ops": sum(a.ops_count for a in AGENTS)}

twin_router = APIRouter(tags=["twin"])
@twin_router.get("/twin")
def get_twins():
    twins = [DigitalTwin(a) for a in AGENTS]
    return {"twins": [t.to_dict() for t in twins], "count": len(twins)}
@twin_router.get("/twin/{agent_name}")
def get_twin_detail(agent_name: str):
    agent = next((a for a in AGENTS if a.name.lower() == agent_name.lower()), None)
    if not agent:
        return {"error": "Agent not found"}
    twin = DigitalTwin(agent)
    return {"twin": twin.to_dict(), "predictions": twin.predict_next(steps=5)}

app = FastAPI(title="GENESIS V14", version="14.0.0")
app.include_router(health_router)
app.include_router(agent_router)
app.include_router(twin_router)

@app.get("/status")
def status():
    for ag in AGENTS: ag.process()
    fe = FREE_ENERGY.update(np.random.randn(64))
    return {"system": "GENESIS V14", "status": "operational", "version": "14.0.0", "boot_time": BOOT_TIME, "agents_online": len([a for a in AGENTS if a.status == "online"]), "total_ops": sum(a.ops_count for a in AGENTS), "total_spikes": sum(a.neuron.spikes for a in AGENTS), "free_energy": round(fe, 4), "cpu_percent": psutil.cpu_percent(), "memory_mb": round(psutil.virtual_memory().used / 1e6, 1), "compute_mode": os.getenv("GENESIS_COMPUTE", "sparse")}

@app.get("/apps")
def get_apps():
    return {"engines": ["SNN", "HDC", "VSC", "FreeEnergy", "DigitalTwin"], "agents": len(AGENTS), "version": "14.0.0"}

@app.get("/deploy")
def get_deploy():
    return {"targets": DEPLOY_TARGETS, "last_deploy": BOOT_TIME}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    rows = ""
    for a in AGENTS:
        a.process()
        rows += "<tr><td>" + a.name + "</td><td>" + a.role + "</td><td>" + a.status + "</td><td>" + str(a.ops_count) + "</td><td>" + str(a.neuron.spikes) + "</td></tr>"
    total_ops = sum(a.ops_count for a in AGENTS)
    fe = FREE_ENERGY.update(np.random.randn(64))
    tgt = ""
    for name, info in DEPLOY_TARGETS.items():
        tgt += "<tr><td>" + name + "</td><td>" + info["status"] + "</td></tr>"
    twins_rows = ""
    for a in AGENTS:
        t = DigitalTwin(a)
        twins_rows += "<tr><td>" + t.agent_name + "</td><td>" + str(t.ops_snapshot) + "</td><td>" + str(t.spikes_snapshot) + "</td><td>" + t.snapshot_time + "</td></tr>"
    html = '<!DOCTYPE html><html><head><title>GENESIS V14</title><style>body{font-family:monospace;background:#0a0a0a;color:#0f0;margin:20px}h1{color:#0ff;text-align:center}h2{color:#0ff}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #333;padding:8px;text-align:left}th{background:#111;color:#0ff}.card{background:#111;border:1px solid #333;padding:15px;margin:10px 0;border-radius:8px}.metric{font-size:2em;color:#0ff}a{color:#0f0}</style></head><body><h1>GENESIS V14 - Ultra-Meta Cognitive OS</h1><div class="card"><span class="metric">' + str(total_ops) + '</span> Total OPS | <span class="metric">' + str(round(fe,2)) + '</span> Free Energy | <span class="metric">' + str(len(AGENTS)) + '</span> Agents</div><h2>Agents</h2><table><tr><th>Name</th><th>Role</th><th>Status</th><th>OPS</th><th>Spikes</th></tr>' + rows + '</table><h2>Digital Twins</h2><table><tr><th>Agent</th><th>OPS Snapshot</th><th>Spikes Snapshot</th><th>Snapshot Time</th></tr>' + twins_rows + '</table><h2>Cloud Targets</h2><table><tr><th>Target</th><th>Status</th></tr>' + tgt + "</table><p>API: <a href='/health'>/health</a> | <a href='/status'>/status</a> | <a href='/agents'>/agents</a> | <a href='/ops'>/ops</a> | <a href='/twin'>/twin</a> | <a href='/deploy'>/deploy</a> | <a href='/docs'>/docs</a></p></body></html>"
    return html

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
