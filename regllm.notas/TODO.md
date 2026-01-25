  1. Investigate larger sequence length (your priority item - currently 512)                                                                                                            
  2. Add Unsloth integration - can provide 2x+ speedup                                                                                                                                  
  3. Enable Gradient Checkpointing - reduces VRAM usage                                                                                                                                 
  4. Configure Flash Attention 2 - faster attention computation                                                                                                                         
  5. Switch to Paged AdamW 8-bit - reduces optimizer memory  


  ⎿  ☐ Investigate using larger sequence length (currently 512)                                                                                                                         
     ☐ Add Unsloth integration for faster training                                                                                                                                      
     ☐ Enable Gradient Checkpointing in TrainingArguments                                                                                                                               
     ☐ Configure Flash Attention 2                                                                                                                                                      
     ☐ Switch to Paged AdamW 8-bit optimizer                                                                                                                                            
     ☐ Integrate Weights & Biases (wandb) for training visualization 
- [ ] Add realtime feedback from  users, RHLF