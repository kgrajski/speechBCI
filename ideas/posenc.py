   @staticmethod
    def _gen_sincos_position_encoding(sample, mask, max_input_seq_len=512):
        """
        Generate a positional encoding for the sample data.
        Apply the sinusoidal positional encoding to the sample data.

        Args:
            sample (torch.Tensor): The sample data tensor.
            mask (torch.Tensor): The attention mask tensor.

        Returns:
            torch.Tensor: The positional encoding tensor.
        """
            # Generate sinusoidal positional encoding
        print(type(sample))
        emb_dim = sample.shape[1]  # Get embedding dimension from sample
        position_enc = torch.zeros(max_input_seq_len, emb_dim)
        position = torch.arange(0, max_input_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        
            # Handle both even and odd embedding dimensions
        position_enc[:, 0::2] = torch.sin(position * div_term[:emb_dim//2])
        if emb_dim % 2 == 0:
            position_enc[:, 1::2] = torch.cos(position * div_term[:emb_dim//2])
        else:
            position_enc[:, 1::2] = torch.cos(position * div_term[:(emb_dim-1)//2])
        
            # Apply position encoding only to actual content (not padding)
        position_enc = position_enc * mask.unsqueeze(1).float()  
        
            # For initial testing, generate a plot of the positional encoding
        import plotly.express as px

                    # Create a DataFrame from the position encoding tensor
        df = position_enc.cpu().numpy()
        
        fig = px.line(df, title='Positional Encoding', labels={'index': 'Position', 'value': 'Encoding Value'})
        fig.update_layout(showlegend=False)
        fig.write_html("positional_encoding.html")
                
        return position_enc
    
    