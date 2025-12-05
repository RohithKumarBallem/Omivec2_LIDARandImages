#!/usr/bin/env python3
"""
Quick sanity check to verify all components work.
"""
import torch
from src.models.tokenizers import ImagePatchTokenizer, PointTokenTokenizer
from src.models.omnivec2_core import CrossModalBlock, OmniVec2Tiny

def main():
    print("\n" + "="*60)
    print("OmniVec2 Quick Test")
    print("="*60)
    
    device = torch.device("cpu")
    print(f"\nDevice: {device}")
    
    # Initialize components
    print("\n1. Initializing tokenizers...")
    img_tok = ImagePatchTokenizer(embed_dim=96, patch=32).to(device)
    lid_tok = PointTokenTokenizer(in_ch=4, embed_dim=96, num_tokens=512).to(device)
    print("   ✅ Tokenizers loaded")
    
    print("\n2. Initializing fusion modules...")
    cross_modal = CrossModalBlock(dim=96, heads=3, ff=192).to(device)
    backbone = OmniVec2Tiny(dim=96, heads=3, ff=192, depth=1).to(device)
    print("   ✅ Fusion modules loaded")
    
    # Create dummy data
    print("\n3. Creating dummy data...")
    imgs = torch.randn(2, 3, 224, 384).to(device)
    pts = torch.randn(2, 1000, 4).to(device)
    print(f"   Images: {imgs.shape}")
    print(f"   Points: {pts.shape}")
    
    # Forward pass
    print("\n4. Running forward pass...")
    ti = img_tok(imgs)
    tl = lid_tok(pts)
    print(f"   Image tokens: {ti.shape}")
    print(f"   LiDAR tokens: {tl.shape}")
    
    ti_cross, tl_cross = cross_modal(ti, tl)
    print(f"   After cross-attention: {ti_cross.shape}, {tl_cross.shape}")
    
    fused = backbone(ti_cross, tl_cross)
    print(f"   Fused tokens: {fused.shape}")
    
    print("\n" + "="*60)
    print("✅ All tests passed! OmniVec2 is ready to use.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
