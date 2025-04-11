    
    
        # Run diagnostics if enabled (every epoch)
        if diagnostics:
            diagnostics.capture_epoch_diagnostics(model, train_dl, epoch)
            
            # Add more specific diagnostics everyt few epochs
            if epoch % 4 == 0: # At the outset, do a diagnostic each epoch
                with torch.no_grad():
                    # Get sample batch for detailed analysis
                    sample_batch = next(iter(train_dl))
                    inputs = sample_batch["vqvae_embeddings"].to(device)
                    
                    # Analyze adapter outputs directly
                    adapter_outputs = adapter(inputs)
                    diagnostics._analyze_adapter_outputs(adapter_outputs, epoch)
                    
                    # Visualize attention if appropriate epoch
                    if epoch % 5 == 0:
                        diagnostics._create_adapter_output_visualization(adapter_outputs)

    
    # Final comprehensive diagnostics after training
    if diagnostics:
        print("\nGenerating comprehensive diagnostics report...")
        diagnostics.run_all_diagnostics(val_dl, model)
        
        # Generate visualizations of attention patterns
        diagnostics.visualize_attention_patterns()
        
        # Analyze mode collapse over training
        diagnostics.analyze_mode_collapse()
        
            # Model-specific parameters unchanged...
    model_type = model_type.lower()
    if model_type == "mbart":
        generation_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["en_XX"]
    elif model_type == "t5":
        generation_kwargs["decoder_start_token_id"] = (
            model.t5_model.config.decoder_start_token_id
        )
    elif model_type == "bart":
        generation_kwargs["decoder_start_token_id"] = tokenizer.bos_token_id
        
        
        
    if enable_diagnostics:
        model_diagnostics = ModelDiagnostics(
            model=mm_llm,
            adapter=adapter,
            tokenizer=tokenizer,
            writer=writer,  # Pass the existing writer
            output_dir=os.path.join(mmllm_model_dir, "diagnostics"),
        )
    else:
        model_diagnostics = None