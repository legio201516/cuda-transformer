import torch
import torch.nn as nn
import torch.nn.functional as F
import time
B,T,C=32,128,256 # batch, seq_len, d_model
d_ff=4*C
n_heads=4
head_dim=C//n_heads # 32 here

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)

        # attention : Q, K, V projections + output proj

        self.qkv=nn.Linear(C, 3*C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)  # Linear(C, C, bias=False)

        #FFN
        self.ffn= nn.Sequential(
        nn.Linear(C, d_ff),
        nn.GELU(),
        nn.Linear(d_ff,C))


    def forward(self,x):
        # pre-norm + attention + résiduel
        x=x+self.attention(self.ln1(x))
        # pre-norm + ffn + résiduel
        x=x+self.ffn(self.ln2(x))
        return x
    def attention(self, x): 
        B,T,C=x.shape 
        #project Q,K,V all at once and split : 
        qkv=self.qkv(x) # size is (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1) # size is each (B, T, C)

        # heads
        # reshape for multi-head : (B, n_heads, T, head_dim)
        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

        #sqrt(head_dim)
        scale=head_dim**-0.5
        #QKt*scale
        scores = torch.matmul(q,k.transpose(-2,-1))*scale

        #softmax
        attn=F.softmax(scores, dim=-1)
        #attn@V
        out = torch.matmul(attn,v)

        #combine all heads' results :
        out = out.transpose(1,2).contiguous().view(B,T,C)


        return out

    
    

def train():
    return
def test():
    return

    


# ---- benchmark ----
block = TransformerBlock().cuda().half()
x = torch.randn(B, T, C, device='cuda', dtype=torch.float16)
print(x.shape)
block.forward(x)
outputs=block.forward(x)
print("after forward :\n")
print(outputs.shape)

#warpup:
for _ in range(20): _ = block(x)
torch.cuda.synchronize()


N = 200
t0 = time.perf_counter()    
for _ in range(N): y = block(x)
torch.cuda.synchronize()
elapsed_time=time.perf_counter()-t0

print(elapsed_time)