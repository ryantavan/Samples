import gradio as gr
import subprocess
import sys
import os
import signal
import psutil
from typing import Generator
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

#########################
# 1. Global Process Management
#########################

running_processes = {
    "cache": None,   # Pre-caching process
    "train": None    # Training process
}

def terminate_process_tree(proc: subprocess.Popen):
    """
    Recursively terminate the specified process and all of its child processes.
    This is useful for accelerator or multi-process scenarios.
    """
    if proc is None:
        return
    try:
        parent_pid = proc.pid
        if parent_pid is None:
            return
        parent = psutil.Process(parent_pid)
        # Terminate all child processes first
        for child in parent.children(recursive=True):
            child.terminate()
        # Finally, terminate the parent process
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"[WARN] Exception occurred in terminate_process_tree: {e}")

def stop_caching():
    """
    Stop the currently running pre-caching child process (cache_latents + cache_text_encoder_outputs).
    """
    if running_processes["cache"] is not None:
        proc = running_processes["cache"]
        if proc.poll() is None:
            terminate_process_tree(proc)
            running_processes["cache"] = None
            return "[INFO] Requested to stop the pre-caching process (terminated all child processes).\n"
        else:
            return "[WARN] Pre-caching process has already finished, no need to stop.\n"
    else:
        return "[WARN] There is currently no pre-caching process running.\n"

def stop_training():
    """
    Stop the currently running training child process.
    """
    if running_processes["train"] is not None:
        proc = running_processes["train"]
        if proc.poll() is None:
            terminate_process_tree(proc)
            running_processes["train"] = None
            return "[INFO] Requested to stop the training process (terminated all child processes).\n"
        else:
            return "[WARN] Training process has already finished, no need to stop.\n"
    else:
        return "[WARN] There is currently no training process running.\n"

#########################
# 2. Process Input Dataset Config Path (supports text or file)
#########################

def get_dataset_config(file_path: str, text_path: str) -> str:
    """
    Based on the user's input of file_path (str, not uploaded) and text_path (str),
    return the final toml path to be used:
    - If file_path is not empty and is a valid file, use file_path first.
    - Otherwise, use text_path.
    - If both are empty, return an empty string.
    """
    if file_path and os.path.isfile(file_path):
        return file_path
    elif text_path.strip():
        return text_path.strip()
    else:
        return ""

#########################
# 3. Pre-caching
#########################

def run_cache_commands(
    dataset_config_file: str,  # Only retrieves the path
    dataset_config_text: str,
    enable_low_memory: bool,
    skip_existing: bool
) -> Generator[str, None, None]:
    """
    Use a generator function with accumulated text to append all output to a single textbox.
    Each line is also printed to the console in real time.
    """
    # Determine the final dataset_config path
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)

    python_executable = "./python_embeded/python.exe"

    # Command for the first stage
    cache_latents_cmd = [
        python_executable, "cache_latents.py",
        "--dataset_config", dataset_config,
        "--vae", "./models/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
        "--vae_chunk_size", "32",
        "--vae_tiling"
    ]
    if enable_low_memory:
        cache_latents_cmd.extend(["--vae_spatial_tile_sample_min_size", "128", "--batch_size", "1"])
    if skip_existing:
        cache_latents_cmd.append("--skip_existing")

    # Command for the second stage
    cache_text_encoder_cmd = [
        python_executable, "cache_text_encoder_outputs.py",
        "--dataset_config", dataset_config,
        "--text_encoder1", "./models/ckpts/text_encoder",
        "--text_encoder2", "./models/ckpts/text_encoder_2",
        "--batch_size", "16"
    ]
    if enable_low_memory:
        cache_text_encoder_cmd.append("--fp8_llm")

def run_and_stream_output(cmd):
    """
    Run a command and stream its output, handling encoding issues by reading binary data
    and safely decoding it.
    """
    accumulated = ""
    
    # Use subprocess.PIPE for output capture but with binary mode (not text mode)
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=False,  # Make sure this is False to read bytes
        bufsize=1,
        encoding='utf-8'
    )
    running_processes["cache"] = process
    
    # Read and process output in binary mode, one byte at a time if needed
    collected_bytes = bytearray()
    
    while True:
        # Read one byte
        byte = process.stdout.read(1)
        if not byte:  # End of stream
            break
            
        collected_bytes.append(byte[0])
        
        # If we have a newline or enough bytes, process what we have
        if byte == b'\n' or len(collected_bytes) > 1024:
            try:
                # Try various encodings with explicit error handling
                line = collected_bytes.decode('utf-8', errors='replace')
            except:
                # Fallback to a very permissive encoding
                line = collected_bytes.decode('latin-1', errors='replace')
            
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated
            collected_bytes = bytearray()
    
    # Process any remaining bytes
    if collected_bytes:
        try:
            line = collected_bytes.decode('utf-8', errors='replace')
        except:
            line = collected_bytes.decode('latin-1', errors='replace')
        
        print(line, end="", flush=True)
        accumulated += line
        yield accumulated
    
    return_code = process.wait()
    running_processes["cache"] = None
    
    if return_code != 0:
        error_msg = f"\n[ERROR] Command execution failed with return code: {return_code}\n"
        accumulated += error_msg
        yield accumulated

    # Run the first command
    accumulated_main = "\n[INFO] Starting first pre-caching stage (cache_latents.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_latents_cmd):
        yield content
    accumulated_main += "\n[INFO] First pre-caching stage completed.\n"
    yield accumulated_main

    # Run the second command
    accumulated_main += "\n[INFO] Starting second pre-caching stage (cache_text_encoder_outputs.py)...\n\n"
    yield accumulated_main
    for content in run_and_stream_output(cache_text_encoder_cmd):
        yield content
    accumulated_main += "\n[INFO] Second pre-caching stage completed.\n"
    yield accumulated_main

#########################
# 4. Training
#########################

def run_training(
    dataset_config_file: str,
    dataset_config_text: str,
    max_train_epochs: int,
    learning_rate: str,
    network_dim: int,
    network_alpha: int,
    gradient_accumulation_steps: int,
    enable_low_vram: bool,
    blocks_to_swap: int,
    output_dir: str,
    output_name: str,
    save_every_n_epochs: int
) -> Generator[str, None, None]:
    """
    The training command uses an accumulated approach so that all output is appended,
    and the process can be stopped.
    """
    dataset_config = get_dataset_config(dataset_config_file, dataset_config_text)

    python_executable = "./python_embeded/python.exe"
    command = [
        python_executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        "hv_train_network.py",
        "--dit", "models/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        "--dataset_config", dataset_config,
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", learning_rate,
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module=networks.lora",
        f"--network_dim={network_dim}",
        f"--network_alpha={network_alpha}",
        "--timestep_sampling", "sigmoid",
        "--discrete_flow_shift", "1.0",
        "--max_train_epochs", str(max_train_epochs),
        "--seed", "42",
        "--output_dir", output_dir,
        "--output_name", output_name,
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        "--logging_dir", "./log",
        "--log_with", "tensorboard",
        "--save_every_n_epochs", str(save_every_n_epochs)
    ]

    if enable_low_vram:
        command.extend(["--blocks_to_swap", str(blocks_to_swap)])

    def run_and_stream_output(cmd):
        accumulated = ""
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        running_processes["train"] = process

        for line in process.stdout:
            print(line, end="", flush=True)
            accumulated += line
            yield accumulated

        return_code = process.wait()
        running_processes["train"] = None

        if return_code != 0:
            error_msg = f"\n[ERROR] Command execution failed with return code: {return_code}\n"
            accumulated += error_msg
            yield accumulated

    start_message = "[INFO] Starting training command...\n\n"
    yield start_message
    accumulated_main = start_message

    for content in run_and_stream_output(command):
        yield content
        accumulated_main = content  # The final content contains all the logs

    accumulated_main += "\n[INFO] Training command execution completed.\n"
    yield accumulated_main

#########################
# 5. LoRA Conversion (Page 3)
#########################

def run_lora_conversion(lora_file_path: str, output_dir: str) -> Generator[str, None, None]:
    """
    - The user only selects a path, does not upload a file.
    - Command: python convert_lora.py --input <in> --output <out> --target other
    - The output file is named: original_filename + "_converted.safetensors"
    """
    if not lora_file_path or not os.path.isfile(lora_file_path):
        yield "[ERROR] No valid LoRA file path selected\n"
        return

    python_executable = "./python_embeded/python.exe"

    # Get the file name from lora_file_path
    in_path = lora_file_path  # local path
    basename = os.path.basename(in_path)  # e.g. rem_lora.safetensors
    filename_no_ext, ext = os.path.splitext(basename)
    # Construct new file name
    out_name = f"{filename_no_ext}_converted{ext}"  # e.g. rem_lora_converted.safetensors

    # If output_dir is empty, default to the current directory
    if not output_dir.strip():
        output_dir = "."

    # Join to create the full output path
    out_path = os.path.join(output_dir, out_name)

    command = [
        python_executable, "convert_lora.py",
        "--input", in_path,
        "--output", out_path,
        "--target", "other"
    ]

    accumulated = ""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

    for line in process.stdout:
        print(line, end="", flush=True)
        accumulated += line
        yield accumulated

    return_code = process.wait()
    if return_code != 0:
        error_msg = f"\n[ERROR] convert_lora.py failed to run with return code: {return_code}\n"
        accumulated += error_msg
        yield accumulated
    else:
        msg = f"\n[INFO] LoRA conversion completed! Output file: {out_path}\n"
        accumulated += msg
        yield accumulated


#########################
# Build Gradio UI
#########################
with gr.Blocks() as demo:
    gr.Markdown("# AI Software Musubi Tuner GUI - Code from Kohya, GUI by TTP")

    ########################################
    # (1) Pre-caching Page
    ########################################
    with gr.Tab("Pre-caching"):
        gr.Markdown("## Latent and Text Encoder Output Pre-caching")

        with gr.Row():
            dataset_config_file_cache = gr.File(
                label="Browse and select dataset_config (toml)",
                file_count="single",
                file_types=[".toml"],
                type="filepath",  # Only return local path
            )
            dataset_config_text_cache = gr.Textbox(
                label="Or enter toml path manually",
                placeholder="K:/ai_software/musubi-tuner/train/test/test.toml"
            )

        enable_low_memory = gr.Checkbox(
            label="Enable Low Memory Mode",
            value=False
        )
        skip_existing = gr.Checkbox(
            label="Skip existing cache files (--skip_existing)",
            value=False
        )

        with gr.Row():
            run_cache_button = gr.Button("Run Pre-caching")
            stop_cache_button = gr.Button("Stop Pre-caching")

        cache_output = gr.Textbox(
            label="Pre-caching Output",
            lines=20,
            interactive=False
        )

        run_cache_button.click(
            fn=run_cache_commands,
            inputs=[dataset_config_file_cache, dataset_config_text_cache, enable_low_memory, skip_existing],
            outputs=cache_output
        )

        stop_cache_button.click(
            fn=stop_caching,
            inputs=None,
            outputs=cache_output
        )

    ########################################
    # (2) Training Page
    ########################################
    with gr.Tab("Training"):
        gr.Markdown("## HV Network Training")

        with gr.Row():
            dataset_config_file_train = gr.File(
                label="Browse and select dataset_config (toml)",
                file_count="single",
                file_types=[".toml"],
                type="filepath",  # Only return local path
            )
            dataset_config_text_train = gr.Textbox(
                label="Or enter toml path manually",
                placeholder="K:/ai_software/musubi-tuner/train/test/test.toml"
            )

        with gr.Row():
            max_train_epochs = gr.Number(
                label="Number of Training Epochs (>=2)",
                value=16,
                precision=0
            )
            learning_rate = gr.Textbox(
                label="Learning Rate (e.g., 1e-4)",
                value="1e-4"
            )

        with gr.Row():
            network_dim = gr.Number(
                label="Training Dim (2-128)",
                value=32,
                precision=0
            )
            network_alpha = gr.Number(
                label="Training Alpha (1-128)",
                value=16,
                precision=0
            )

        with gr.Row():
            gradient_accumulation_steps = gr.Number(
                label="Gradient Accumulation Steps (recommend even number)",
                value=1,
                precision=0
            )
            enable_low_vram = gr.Checkbox(
                label="Enable Low VRAM Mode",
                value=False
            )

        blocks_to_swap = gr.Number(
            label="Blocks to Swap (20-36, even number)",
            value=20,
            precision=0,
            visible=False
        )

        def toggle_blocks_swap(checked):
            return gr.update(visible=checked)

        enable_low_vram.change(
            toggle_blocks_swap,
            inputs=enable_low_vram,
            outputs=blocks_to_swap
        )

        with gr.Row():
            output_dir_input = gr.Textbox(
                label="Output Directory",
                value="./output",
                placeholder="./output"
            )
            output_name_input = gr.Textbox(
                label="Output Name (e.g., rem_test)",
                value="lora",
                placeholder="rem_test"
            )

        with gr.Row():
            save_every_n_epochs = gr.Number(
                label="Save every n epochs (save_every_n_epochs)",
                value=1,
                precision=0
            )

        with gr.Row():
            run_train_button = gr.Button("Run Training")
            stop_train_button = gr.Button("Stop Training")

        train_output = gr.Textbox(
            label="Training Output",
            lines=20,
            interactive=False
        )

        run_train_button.click(
            fn=run_training,
            inputs=[
                dataset_config_file_train,
                dataset_config_text_train,
                max_train_epochs,
                learning_rate,
                network_dim,
                network_alpha,
                gradient_accumulation_steps,
                enable_low_vram,
                blocks_to_swap,
                output_dir_input,
                output_name_input,
                save_every_n_epochs
            ],
            outputs=train_output
        )

        stop_train_button.click(
            fn=stop_training,
            inputs=None,
            outputs=train_output
        )

    ########################################
    # (3) LoRA Conversion Page
    ########################################
    with gr.Tab("LoRA Conversion"):
        gr.Markdown("## Convert LoRA to another format (compatible with ComfyUI)")

        lora_file_input = gr.File(
            label="Select Musubi LoRA file (.safetensors) (path only)",
            file_count="single",
            file_types=[".safetensors"],
            type="filepath"  # Only return local path
        )
        output_dir_conversion = gr.Textbox(
            label="Output Directory (optional); if empty, defaults to current directory",
            value="./output",
            placeholder="K:/ai_software/musubi-tuner/converted_output"
        )

        convert_button = gr.Button("Convert LoRA")
        conversion_output = gr.Textbox(
            label="Conversion Output Log",
            lines=15,
            interactive=False
        )

        convert_button.click(
            fn=run_lora_conversion,
            inputs=[lora_file_input, output_dir_conversion],
            outputs=conversion_output
        )

    ########################################
    # Notes
    ########################################
    gr.Markdown("""
### Notes
1. **Path Format**: Please use the correct path format (Windows can use forward slashes `/` or escaped backslashes `\\`).
2. **LoRA Conversion**: Enter the `.safetensors` path (do not upload the file). The output file will automatically have `_converted.safetensors` appended to its name.
    """)

demo.queue()
demo.launch()
