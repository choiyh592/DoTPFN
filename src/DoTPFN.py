# Add imports

# Modify ModalityProjector

class DoTPFN(nn.Module):
    def __init__(self, pretrained_tabpfn: nn.Module, non_tab_dim: int, n_heads: int = 32, k_tokens: int = 24):
        super().__init__()
        # Ensure we have the base PyTorch architecture, not the Sklearn wrapper
        self.tabpfn = pretrained_tabpfn 
        self.d_model = self.tabpfn.input_size
        
        # Initialize our trainable Modality Projector
        self.modality_projector = ModalityProjector(
            non_tab_dim=non_tab_dim, 
            d_model=self.d_model, 
            n_heads=n_heads, 
            k_tokens=k_tokens
        )
        
        self._freeze_tabular_encoders()
        
    def _freeze_tabular_encoders(self):
        """Freezes the data processing components, keeps the Transformer trainable."""
        # 1. Freeze Tabular Encoders
        modules_to_freeze = [
            self.tabpfn.feature_group_embedder,
            self.tabpfn.target_embedder,
            self.tabpfn.feature_positional_embedding_embeddings
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
                
        # 2. Ensure Backbone and Decoder are Trainable
        modules_to_train = [
            self.tabpfn.blocks,
            self.tabpfn.output_projection,
            self.tabpfn.add_thinking_rows # Trainable thinking tokens
        ]
        for module in modules_to_train:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, x_tab: torch.Tensor, y: torch.Tensor, x_non_tab: torch.Tensor):
        """
        Args:
            x_tab: Tabular data tensor [Ri, Batch, Features]
            y: Labels tensor [Rt] or [Rt, Batch]
            x_non_tab: Extracted [CLS] embeddings [Batch, Ri, non_tab_dim]
        """
        num_train_rows, batch_size, _ = x_tab.shape
        num_train_labels = y.shape[0]
        
        # ==========================================
        # Step 1: FROZEN Tabular & Target Processing
        # ==========================================
        with torch.no_grad():
            # Output Shape: [B, Ri, G, d_model] (where G is number of feature groups)
            tab_tokens = self.tabpfn._preprocess_and_embed_features(
                x_RiBC=x_tab, 
                num_train_labels=num_train_labels, 
                batch_size=batch_size
            )
            
            # Output Shape: [B, Ri, d_model]
            target_tokens = self.tabpfn._preprocess_and_embed_targets(
                y=y, 
                num_train_rows=num_train_rows, 
                num_train_labels=num_train_labels, 
                batch_size=batch_size
            )
            
        # Combine tabular features with targets as an extra "column"
        # Shape: [B, Ri, G + 1, d_model]
        tab_and_target_tokens = torch.cat((tab_tokens, target_tokens.unsqueeze(2)), dim=2)
        
        # ==========================================
        # Step 2: TRAINABLE Modality Projection
        # ==========================================
        # Output Shape: [B, Ri, K, d_model]
        non_tab_tokens = self.modality_projector(x_non_tab)
        
        # ==========================================
        # Step 3: Modality Fusion
        # ==========================================
        # Concatenate along the "column/feature" dimension
        # Shape: [B, Ri, G + 1 + K, d_model]
        fused_tokens = torch.cat((tab_and_target_tokens, non_tab_tokens), dim=2)
        
        # ==========================================
        # Step 4: TRAINABLE TabPFN Backbone
        # ==========================================
        x_BRCD, num_train_and_thinking_rows = self.tabpfn.add_thinking_rows(
            fused_tokens, 
            single_eval_pos=num_train_labels
        )
        
        # In-context learning via the Transformer Stack
        for block in self.tabpfn.blocks:
            x_BRCD = block(
                x_BRCD, 
                single_eval_pos=num_train_and_thinking_rows, 
                save_peak_memory_factor=None # Standard backprop setup
            )
            
        # ==========================================
        # Step 5: TRAINABLE Decoder
        # ==========================================
        # Extract the test predictions from the final column (-1) 
        test_embeddings_BMD = x_BRCD[:, num_train_and_thinking_rows:, -1]
        test_embeddings_MBD = test_embeddings_BMD.transpose(0, 1)
        test_output_MB1 = self.tabpfn.output_projection(test_embeddings_MBD)
        
        return test_output_MB1