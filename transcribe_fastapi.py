import argparse
import gc
import os
import io
import warnings

import numpy as np
import torch

from .alignment import align, load_align_model
from .asr import load_model
from .audio import load_audio
from .diarize import DiarizationPipeline, assign_word_speakers
from .types import AlignedTranscriptionResult, TranscriptionResult
from .utils import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
    get_writer,
    optional_float,
    optional_int,
    str2bool,
)


def load_arguments():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # audio file(s) to transcribe, required, but only if not running as a daemon.
    parser.add_argument("audio", nargs="*", default=[], help="audio file(s) to transcribe")

    parser.add_argument("--model", default="small", help="name of the Whisper model to use")
    parser.add_argument("--model_cache_only", type=str2bool, default=False,
                        help="If True, will not attempt to download models, instead using cached models from --model_dir")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use for PyTorch inference")
    parser.add_argument("--device_index", default=0, type=int,
                        help="device index to use for FasterWhisper inference")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="the preferred batch size for inference")
    parser.add_argument("--compute_type", default="float16", type=str,
                        choices=["float16", "float32", "int8"], help="compute type for computation")

    parser.add_argument("--output_dir", "-o", type=str, default=".",
                        help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all",
                        choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"],
                        help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe",
                        choices=["transcribe", "translate"],
                        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, specify None to perform language detection")

    # alignment params
    parser.add_argument("--align_model", default=None,
                        help="Name of phoneme-level ASR model to do alignment")
    parser.add_argument("--interpolate_method", default="nearest",
                        choices=["nearest", "linear", "ignore"],
                        help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    parser.add_argument("--no_align", action='store_true',
                        help="Do not perform phoneme alignment")
    parser.add_argument("--return_char_alignments", action='store_true',
                        help="Return character-level alignments in the output json file")

    # vad params
    parser.add_argument("--vad_method", type=str, default="pyannote",
                        choices=["pyannote", "silero"], help="VAD method to be used")
    parser.add_argument("--vad_onset", type=float, default=0.500,
                        help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    parser.add_argument("--vad_offset", type=float, default=0.363,
                        help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    parser.add_argument("--chunk_size", type=int, default=30,
                        help="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.")

    # diarization params
    parser.add_argument("--diarize", action="store_true",
                        help="Apply diarization to assign speaker labels to each segment/word")
    parser.add_argument("--min_speakers", default=None, type=int,
                        help="Minimum number of speakers to in audio file")
    parser.add_argument("--max_speakers", default=None, type=int,
                        help="Maximum number of speakers to in audio file")

    parser.add_argument("--temperature", type=float, default=0,
                        help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5,
                        help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5,
                        help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=1.0,
                        help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144")
    parser.add_argument("--suppress_tokens", type=str, default="-1",
                        help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--suppress_numerals", action="store_true",
                        help="whether to suppress numeric symbols and currency symbols during sampling")
    parser.add_argument("--initial_prompt", type=str, default=None,
                        help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False,
                        help="if True, provide the previous output of the model as a prompt for the next window")
    parser.add_argument("--fp16", type=str2bool, default=True,
                        help="whether to perform inference in fp16; True by default")
    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2,
                        help="temperature to increase when falling back")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4,
                        help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0,
                        help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6,
                        help="if the probability of the <|nospeech|> token is higher than this value, consider the segment as silence")
    parser.add_argument("--max_line_width", type=optional_int, default=None,
                        help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None,
                        help="(not possible with --no_align) the maximum number of lines in a segment")
    parser.add_argument("--highlight_words", type=str2bool, default=False,
                        help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--segment_resolution", type=str, default="sentence",
                        choices=["sentence", "chunk"],
                        help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--threads", type=optional_int, default=0,
                        help="number of threads used by torch for CPU inference")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face Access Token to access PyAnnote gated models")
    parser.add_argument("--print_progress", type=str2bool, default=False,
                        help="if True, progress will be printed in transcribe() and align() methods.")

    # New daemon and server parameters.
    parser.add_argument("--daemon", action="store_true",
                        help="Run as a daemon with FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host for the FastAPI server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the FastAPI server")
    # fmt: on

    # If not running as a daemon, require at least one audio file.
    args = vars(parser.parse_args())
    if not args["daemon"] and not args["audio"]:
        parser.error("At least one audio file must be provided.")

    return args


def load_pipelines(args):
    """
    Load and return all pipelines (models) based on the CLI arguments.
    This includes the ASR model, the alignment model (if enabled), and the diarization pipeline (if enabled).
    """
    # Build a configuration dictionary from args.
    config = {}
    config["model_name"] = args["model"]
    config["batch_size"] = args["batch_size"]
    config["model_dir"] = args["model_dir"]
    config["model_cache_only"] = args["model_cache_only"]
    config["device"] = args["device"]
    config["device_index"] = args["device_index"]
    config["compute_type"] = args["compute_type"]
    config["verbose"] = args["verbose"]
    config["task"] = args["task"]
    config["language"] = args["language"]
    config["align_model"] = args["align_model"]
    config["interpolate_method"] = args["interpolate_method"]
    config["no_align"] = args["no_align"]
    config["return_char_alignments"] = args["return_char_alignments"]
    config["hf_token"] = args["hf_token"]
    config["vad_method"] = args["vad_method"]
    config["vad_onset"] = args["vad_onset"]
    config["vad_offset"] = args["vad_offset"]
    config["chunk_size"] = args["chunk_size"]
    config["diarize"] = args["diarize"]
    config["min_speakers"] = args["min_speakers"]
    config["max_speakers"] = args["max_speakers"]

    # Sampling/decoding parameters.
    config["temperature"] = args["temperature"]
    config["temperature_increment_on_fallback"] = args["temperature_increment_on_fallback"]
    config["beam_size"] = args["beam_size"]
    config["patience"] = args["patience"]
    config["length_penalty"] = args["length_penalty"]
    config["compression_ratio_threshold"] = args["compression_ratio_threshold"]
    config["log_prob_threshold"] = args["logprob_threshold"]
    config["no_speech_threshold"] = args["no_speech_threshold"]
    # Convert suppress_tokens into a list of ints.
    config["suppress_tokens"] = [int(x) for x in args["suppress_tokens"].split(",")]
    config["suppress_numerals"] = args["suppress_numerals"]
    config["initial_prompt"] = args["initial_prompt"]
    config["condition_on_previous_text"] = args["condition_on_previous_text"]
    config["fp16"] = args["fp16"]
    config["threads"] = args["threads"]

    # Process temperature into a tuple or list.
    temp = config["temperature"]
    inc = config["temperature_increment_on_fallback"]
    if inc is not None:
        config["temperatures"] = tuple(np.arange(temp, 1.0 + 1e-6, inc))
    else:
        config["temperatures"] = [temp]

    # Process language.
    if config["language"] is not None:
        config["language"] = config["language"].lower()
        if config["language"] not in LANGUAGES:
            if config["language"] in TO_LANGUAGE_CODE:
                config["language"] = TO_LANGUAGE_CODE[config["language"]]
            else:
                raise ValueError(f"Unsupported language: {config['language']}")
    if config["model_name"].endswith(".en") and config["language"] != "en":
        if config["language"] is not None:
            warnings.warn(f"{config['model_name']} is an English-only model but received '{config['language']}'; using English instead.")
        config["language"] = "en"
    config["align_language"] = config["language"] if config["language"] is not None else "en"

    # Set up threads.
    faster_whisper_threads = 4
    if config["threads"] > 0:
        torch.set_num_threads(config["threads"])
        faster_whisper_threads = config["threads"]
    config["faster_whisper_threads"] = faster_whisper_threads

    # Build ASR options.
    asr_options = {
        "beam_size": config["beam_size"],
        "patience": config["patience"],
        "length_penalty": config["length_penalty"],
        "temperatures": config["temperatures"],
        "compression_ratio_threshold": config["compression_ratio_threshold"],
        "log_prob_threshold": config["log_prob_threshold"],
        "no_speech_threshold": config["no_speech_threshold"],
        "condition_on_previous_text": False,
        "initial_prompt": config["initial_prompt"],
        "suppress_tokens": config["suppress_tokens"],
        "suppress_numerals": config["suppress_numerals"],
    }
    config["asr_options"] = asr_options

    # Load the ASR pipeline.
    asr_model = load_model(
        config["model_name"],
        device=config["device"],
        device_index=config["device_index"],
        download_root=config["model_dir"],
        compute_type=config["compute_type"],
        language=config["language"],
        asr_options=asr_options,
        vad_method=config["vad_method"],
        vad_options={
            "chunk_size": config["chunk_size"],
            "vad_onset": config["vad_onset"],
            "vad_offset": config["vad_offset"],
        },
        task=config["task"],
        local_files_only=config["model_cache_only"],
        threads=faster_whisper_threads,
    )
    pipelines = {"asr": asr_model}

    # Load alignment pipeline if enabled.
    if not config["no_align"]:
        align_model, align_metadata = load_align_model(
            config["align_language"], config["device"], model_name=config["align_model"]
        )
        pipelines["align"] = align_model
        pipelines["align_metadata"] = align_metadata
    else:
        pipelines["align"] = None
        pipelines["align_metadata"] = None

    # Load diarization pipeline if diarization is enabled.
    if config["diarize"]:
        pipelines["diarize"] = DiarizationPipeline(use_auth_token=config["hf_token"], device=config["device"])
    else:
        pipelines["diarize"] = None

    # Save the configuration inside pipelines for later use.
    pipelines["config"] = config
    return pipelines


def process_file(audio_path: str, args, pipelines, mem_verbose=False, return_result=False):
    """
    Process each audio file sequentially: perform transcription,
    then (if enabled) alignment, then (if enabled) diarization, and finally write output.
    
    If mem_verbose is True, print CUDA memory diagnostics after each file is processed.
    """
    config = pipelines["config"]
    writer = get_writer(args["output_format"], args["output_dir"])
    # The writer options are taken from these keys.
    word_options = ["highlight_words", "max_line_count", "max_line_width"]
    writer_args = {key: args[key] for key in word_options if key in args}

    print(">> Processing file:", audio_path)
    audio = load_audio(audio_path)

    # Transcription.
    print(">> Performing transcription on", audio_path)
    result: TranscriptionResult = pipelines["asr"].transcribe(
        audio,
        batch_size=config["batch_size"],
        chunk_size=config["chunk_size"],
        print_progress=args["print_progress"],
        verbose=config["verbose"],
    )

    # Alignment.
    if not config["no_align"]:
        if pipelines["align"] is not None and len(result["segments"]) > 0:
            # Check if the language detected is different from the alignment model language.
            if result.get("language", "en") != pipelines["align_metadata"]["language"]:
                print(f"New language found ({result['language']}); reloading alignment model...")
                pipelines["align"], pipelines["align_metadata"] = load_align_model(
                    result["language"], config["device"]
                )
            print(">> Performing alignment on", audio_path)
            result: AlignedTranscriptionResult = align(
                result["segments"],
                pipelines["align"],
                pipelines["align_metadata"],
                audio,
                config["device"],
                interpolate_method=config["interpolate_method"],
                return_char_alignments=config["return_char_alignments"],
                print_progress=args["print_progress"],
            )

    # Diarization.
    if config["diarize"] and pipelines["diarize"] is not None:
        print(">> Performing diarization on", audio_path)
        diarize_segments = pipelines["diarize"](audio_path, 
                                                min_speakers=config["min_speakers"],
                                                max_speakers=config["max_speakers"])
        result = assign_word_speakers(diarize_segments, result)

    # Set the language field (for consistency in the output).
    result["language"] = config["align_language"]

    # Now, use the writer:
    if return_result:
        # Use an in-memory stream to capture the writer's output.
        out_stream = io.StringIO()
        writer.write_result(result, file=out_stream, options=writer_args)
        formatted = out_stream.getvalue()
    else:
        # In CLI mode, simply write the output.
        writer(result, audio_path, writer_args)
    
    # Clean up after processing this file.
    gc.collect()
    torch.cuda.empty_cache()

    if mem_verbose and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"Memory diagnostics after {audio_path}: allocated {allocated:.1f} MB, reserved {reserved:.1f} MB.")

    # In CLI mode, write output to file; in daemon mode, return the final result.
    if return_result:
        return formatted

def process_files(args, pipelines, mem_verbose=False):
    """
    Process all audio files sequentially by calling process_file on each one (CLI mode).
    """
    for audio_path in args["audio"]:
        process_file(audio_path, args, pipelines, mem_verbose, return_result=False)

def clean_pipelines(pipelines, mem_verbose=False):
    """
    Perform final cleanup of the pipelines, deleting model references
    and clearing CUDA cache.
    """
    # Remove all references to loaded pipelines.
    for key in list(pipelines.keys()):
        del pipelines[key]
    gc.collect()
    torch.cuda.empty_cache()
    if mem_verbose and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"Final cleanup: allocated {allocated:.1f} MB, reserved {reserved:.1f} MB.")


# ---------------- FastAPI Section ----------------

from fastapi import FastAPI, HTTPException, Query
from starlette.concurrency import run_in_threadpool
import asyncio

app = FastAPI(title="WhisperX Transcription Service")

# Global asynchronous lock to allow only one transcription request at a time.
processing_lock = asyncio.Lock()

# Load pipelines once at startup.
# Here we load arguments from the command line. In a production service,
# you might instead load these from environment variables or a config file.
global_args = load_arguments()
global_pipelines = load_pipelines(global_args)


@app.post("/transcribe")
async def transcribe_endpoint(
    filename: str = Query(..., description="Path to the audio file to transcribe"),
    mem_verbose: bool = Query(False, description="Enable memory diagnostics")
):
    """
    RESTful endpoint to process a single audio file.
    Only one such call is allowed at a time to protect GPU resources.
    Instead of writing the output to a file, the final result is returned.
    """
    async with processing_lock:
        try:
            # Run the synchronous process_file() in a thread pool.
            result = await run_in_threadpool(process_file, filename, global_args, global_pipelines, mem_verbose, True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "completed", "filename": filename, "result": result}


@app.on_event("shutdown")
async def shutdown_event():
    clean_pipelines(global_pipelines, mem_verbose=False)


# ---------------- CLI Entrypoint ----------------

def cli(args):
    # Load pipelines (ASR, alignment, diarization) based on CLI arguments.
    pipelines = load_pipelines(args)
    # Process files sequentially (CLI mode writes output to file).
    process_files(args, pipelines, mem_verbose=False)
    # Final cleanup.
    clean_pipelines(pipelines, mem_verbose=False)


if __name__ == "__main__":
    args = load_arguments()
    if args.get("daemon", False):
        # If running as a daemon, start the FastAPI server.
        host = args.pop("host")
        port = args.pop("port")
        import uvicorn
        uvicorn.run(app, host=host, port=port)
    else:
        # When run as a script, run the CLI version.
        cli(args)
