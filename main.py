from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch_geometric.nn.models.schnet
from fastapi.middleware.cors import CORSMiddleware
from torch_geometric.nn import SchNet
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# THE MAGIC BYPASS (Fixes the pyg-lib error)
# ==========================================
def pure_python_radius_graph(pos, r, batch, max_num_neighbors=32, flow='source_to_target', num_workers=1):
    """Calculates 3D atomic distances without needing C++ pyg-lib"""
    # Calculate distance from every atom to every other atom
    dist_matrix = torch.cdist(pos, pos)
    
    # Find atom pairs that are closer than the cutoff radius 'r'
    row, col = torch.where(dist_matrix < r)
    
    # Remove self-loops (atoms connected to themselves)
    mask = row != col
    row, col = row[mask], col[mask]
    
    return torch.stack([row, col], dim=0)

# Overwrite PyG's C++ function with our pure Python function
torch_geometric.nn.models.schnet.radius_graph = pure_python_radius_graph
# ==========================================











app = FastAPI()



# --- ADD THIS CORS BLOCK ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)








# 1. Initialize the architecture natively
device = torch.device("cpu")
model = SchNet(
    hidden_channels=128,
    num_filters=128,
    num_interactions=6,
    num_gaussians=50,
    cutoff=10.0,
    readout='mean'
)

# 2. Load the pure weights (Uses schnet_weights.pt from the previous step)
model.load_state_dict(torch.load("schnet_weights.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()

class MoleculeInput(BaseModel):
    smiles: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "SchNet Inference API (Native PyTorch Bypass Active)"}

@app.post("/predict")
async def predict(data: MoleculeInput):
    # Feature Engineering (3D Coordinates)
    mol = Chem.MolFromSmiles(data.smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    
    conf = mol.GetConformer()
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long, device=device)
    pos = torch.tensor([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=torch.float, device=device)
    batch = torch.zeros(len(z), dtype=torch.long, device=device)
    
    sdf_block = Chem.MolToMolBlock(mol)
    
    # Model Inference
    with torch.no_grad():
        pred = model(z, pos, batch)
        
    return {
        "predicted_logS": round(pred.item(), 3),
        "sdf_file": sdf_block 
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)