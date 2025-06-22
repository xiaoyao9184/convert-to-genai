
import builtins
import logging
import os
import sys
import shutil
import uuid
import re
import contextvars
import requests
import gradio as gr
from huggingface_hub import HfApi, whoami
from onnxruntime_genai.models.builder import create_model
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Callable
from enum import Enum
from tqdm import tqdm
from contextlib import suppress


class ExecutionProvider(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    DML = "dml"
    WEBGPU = "webgpu"
    NVTENSORRT = "NvTensorRtRtx"

class Precision(Enum):
    INT4 = "int4"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"

EXECUTION_PROVIDERS = tuple(x.value for x in ExecutionProvider)

PRECISION_MODES = tuple(x.value for x in Precision)

EXECUTION_PROVIDER_PATH_MAPPING = {
    ExecutionProvider.CPU: "cpu_and_mobile",
    ExecutionProvider.CUDA: "cuda",
    ExecutionProvider.DML: "directml"
}

NAME_EXTRA_OPTIONS_MAPPING = {
    (ExecutionProvider.CPU,Precision.INT4): {
        "rtn-block-32": {
            "int4_algo_config": "rtn",
            "int4_block_size": 32
        },
        "rtn-block-32-acc-level-4": {
            "int4_algo_config": "rtn",
            "int4_block_size": 32,
            "int4_accuracy_level": 4
        }
    },

    (ExecutionProvider.CUDA,Precision.FP16): {
        "": {},
    },
    (ExecutionProvider.CUDA,Precision.INT4): {
        "rtn-block-32": {
            "int4_algo_config": "rtn",
            "int4_block_size": 32
        },
    },

    (ExecutionProvider.DML,Precision.INT4): {
        "awq-block-128": {
            "int4_algo_config": "awq",
            "int4_block_size": 128
        },
    }
}

@dataclass
class Config:
    """Application configuration."""

    _id: Optional[str] = field(default=None, init=False)
    _logger: Optional[logging.Logger] = field(default=None, init=False)
    _logger_path: Optional[Path] = field(default=None, init=False)

    hf_token: str
    hf_username: str
    is_using_user_token: bool
    ignore_converted: bool = False

    hf_base_url: str = "https://huggingface.co"
    output_path: Path = Path("./models")
    cache_path: Path = Path("./cache")
    log_path: Path = Path("./logs")

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables and secrets."""
        system_token = os.getenv("HF_TOKEN")

        if system_token and system_token.startswith("/run/secrets/") and os.path.isfile(system_token):
            with open(system_token, "r") as f:
                system_token = f.read().strip()

        hf_username = (
            os.getenv("SPACE_AUTHOR_NAME") or whoami(token=system_token)["name"]
        )

        output_dir = os.getenv("OUTPUT_DIR") or "./models"
        cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("CACHE_DIR") or "./cache"
        log_dir = os.getenv("LOG_DIR") or "./logs"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True)
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        return cls(
            hf_token=system_token,
            hf_username=hf_username,
            is_using_user_token=False,
            ignore_converted=os.getenv("IGNORE_CONVERTED", "false") == "true",
            output_path=output_path,
            cache_path=cache_path,
            log_path=log_path
        )

    @property
    def id(self):
        if not self._id:
            self._id = str(uuid.uuid4())
        return self._id

    @property
    def logger(self) -> logging.Logger:
        """Get logger."""
        if not self._logger:
            logger = logging.getLogger(self.id)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.FileHandler(self.logger_path)
                handler.setFormatter(logging.Formatter("[%(levelname)s] - %(message)s"))
                logger.addHandler(handler)
            self._logger = logger
        return self._logger

    @property
    def logger_path(self) -> Path:
        """Get logger path."""
        if not self._logger_path:
            logger_path = self.log_path / f"{self.id}.log"
            logger_path.parent.mkdir(exist_ok=True)
            self._logger_path = logger_path
        return self._logger_path

    def token(self, user_token):
        """Update token."""
        if user_token:
            hf_username = whoami(token=user_token)["name"]
        else:
            hf_username = (
                os.getenv("SPACE_AUTHOR_NAME") or whoami(token=self.hf_token)["name"]
            )

        hf_token = user_token or self.hf_token

        if not hf_token:
            raise ValueError(
                "When the user token is not provided, the system token must be set."
            )

        self.hf_token = hf_token
        self.hf_username = hf_username
        self.is_using_user_token = bool(user_token)

class ProgressLogger:
    """Logger with progress update."""

    def __init__(self, logger: logging.Logger, updater: Callable[[int], None]):
        self.logger = logger
        self.updater = updater
        self.last_progress = 1
        self.last_message = None
        self.write_count = 0

    def update(self, percent):
        if percent >= self.last_progress:
            self.updater(percent - self.last_progress)
        else:
            self.updater(self.last_progress - percent)
        self.last_progress = min(self.last_progress, percent)

    def print(self, *args, **kwargs):
        self.last_message = " ".join(str(arg) for arg in args)
        if self.logger:
            self.logger.info(self.last_message.removeprefix("\r"))

        if self.last_message.startswith("\rProgress:"):
            with suppress(Exception):
                percent_str = self.last_message.strip().split()[-1].strip('%')
                percent = float(percent_str)
                self.update(percent)
                self.last_progress = percent

    def write(self, text, write):
        match = re.search(r"pre-uploaded: \d+/\d+ \(([\d.]+)M/([\d.]+)M\)", text)
        if match:
            with suppress(Exception):
                current = float(match.group(1))
                total = float(match.group(2))
                percent = current / total * 100
                self.update(percent)
                self.write_count += 1
        # 60 count for each second
        if self.write_count > 60:
            self.write_count = 0
            write(text)

class ModelConverter:
    """Handles model conversion and upload operations."""

    def __init__(self, config: Config, context: contextvars.ContextVar):
        self.config = config
        self.api = HfApi(token=config.hf_token)
        self.context = context

    def list_tasks(self):
        for execution_provider in EXECUTION_PROVIDERS:
            ep = ExecutionProvider(execution_provider)
            for precision in PRECISION_MODES:
                p = Precision(precision)
                name_extra_options_map = NAME_EXTRA_OPTIONS_MAPPING.get((ep, p), {})
                for name in name_extra_options_map.keys():
                    path_names = [ep.value, p.value]
                    if name:
                        path_names.append(name)
                    path_name = "-".join(path_names)
                    task_name = path_name

                    yield {
                        f"{task_name}": {
                            "🔁 Conversion": "⏳",
                            "📤 Upload": "⏳"
                        }
                    }

    def convert_model(
        self, input_model_id: str, output_model_id: str, progress_updater: Callable[[int], None]
    ) -> Tuple[bool, Optional[str]]:
        """Convert the model to ONNX format."""
        input_dir = ""
        cache_dir = str(self.config.cache_path.absolute())
        output_dir = str(self.config.output_path.absolute() / output_model_id)

        yield f"🧠 Model id: {output_model_id}"

        for execution_provider in (progress_provider := tqdm(EXECUTION_PROVIDERS, disable=False)):
            progress_provider.set_description(f"  Execution provider: {execution_provider}")

            ep = ExecutionProvider(execution_provider)
            path_provider = EXECUTION_PROVIDER_PATH_MAPPING.get(ep, ep.value)

            for precision in (progress_precision := tqdm(PRECISION_MODES, disable=False)):
                progress_precision.set_description(f"    Precision: {precision}")

                p = Precision(precision)
                name_extra_options_map = NAME_EXTRA_OPTIONS_MAPPING.get((ep, p), {})

                for name in (progress_name := tqdm(name_extra_options_map.keys(), disable=False, initial=1, total=len(name_extra_options_map))):
                    progress_name.set_description(f"      Name: {name}")

                    path_names = [ep.value, p.value]
                    if name:
                        path_names.append(name)
                    path_name = "-".join(path_names)
                    task_name = path_name

                    output_path = os.path.join(
                        output_dir,
                        path_provider,
                        path_name,
                    )

                    extra_options = name_extra_options_map[name]
                    extra_options['hf_token'] = "false" if self.config.hf_token == None else self.config.hf_token

                    try:
                        yield {
                            f"{task_name}": {
                                "🔁 Conversion": "🟢"
                            }
                        }
                        self.context.set(ProgressLogger(self.config.logger, progress_updater))
                        for progress_fake in (_ := tqdm(range(100), disable=False)):
                            if progress_fake == 0:
                                create_model(
                                    input_model_id, input_dir, output_path, precision, execution_provider, cache_dir, **extra_options
                                )
                        yield {
                            f"{task_name}": {
                                "🔁 Conversion": "✅"
                            }
                        }
                    except Exception as e:
                        yield {
                            f"{task_name}": {
                                "🔁 Conversion": "❌"
                            }
                        }
                        raise e
        return output_dir

    def upload_model(
        self, input_model_id: str, output_model_id: str, progress_updater: Callable[[int], None]
    ) -> Optional[str]:
        """Upload the converted model to Hugging Face."""
        model_folder_path = self.config.output_path / output_model_id
        hf_model_url = f"{self.config.hf_base_url}/{output_model_id}"

        try:
            self.api.create_repo(output_model_id, exist_ok=True, private=False)
            yield f"🤗 Hugging Face model [{output_model_id}]({hf_model_url})"

            readme_path = f"{model_folder_path}/README.md"
            if not os.path.exists(readme_path):
                with open(readme_path, "w") as file:
                    file.write(self.generate_readme(input_model_id))
            self.context.set(ProgressLogger(self.config.logger, progress_updater))
            self.api.upload_file(
                repo_id=output_model_id,
                path_or_fileobj=readme_path,
                path_in_repo="README.md"
            )
            yield f"🪪 Model card [README.md]({hf_model_url}/blob/main/README.md)"

            for execution_provider in (progress_provider := tqdm(EXECUTION_PROVIDERS, disable=False)):
                ep = ExecutionProvider(execution_provider)
                path_provider = EXECUTION_PROVIDER_PATH_MAPPING.get(ep, ep.value)
                for precision in (progress_precision := tqdm(PRECISION_MODES, disable=False)):
                    p = Precision(precision)
                    name_extra_options_map = NAME_EXTRA_OPTIONS_MAPPING.get((ep, p), {})
                    for name in (progress_name := tqdm(name_extra_options_map.keys(), disable=False, initial=1, total=len(name_extra_options_map))):
                        path_names = [ep.value, p.value]
                        if name:
                            path_names.append(name)
                        path_name = "-".join(path_names)
                        task_name = path_name

                        allow_patterns = os.path.join(
                            path_provider,
                            path_name,
                            "**"
                        )
                        folder_path = str(model_folder_path)

                        try:
                            yield {
                                f"{task_name}": {
                                    "📤 Upload": "🟢"
                                }
                            }
                            self.context.set(ProgressLogger(self.config.logger, progress_updater))
                            for progress_fake in (_ := tqdm(range(100), disable=False)):
                                if progress_fake == 0:
                                    self.api.upload_large_folder(
                                        repo_id=output_model_id, folder_path=folder_path, allow_patterns=allow_patterns,
                                        repo_type="model", print_report_every=1
                                    )
                            yield {
                                f"{task_name}": {
                                    "📤 Upload": "✅"
                                }
                            }
                        except Exception as e:
                            yield {
                                f"{task_name}": {
                                    "📤 Upload": "❌"
                                }
                            }
                            raise e
            return hf_model_url
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(model_folder_path, ignore_errors=True)

    def generate_readme(self, imi: str):
        return (
            "---\n"
            "library_name: onnxruntime-genai\n"
            "base_model:\n"
            f"- {imi}\n"
            "---\n\n"
            f"# {imi.split('/')[-1]} (ONNX Runtime GenAI)\n\n"
            f"This is an ONNX Runtime GenAI version of [{imi}](https://huggingface.co/{imi}). "
            "It was automatically converted and uploaded using "
            "[this space](https://huggingface.co/spaces/xiaoyao9184/convert-to-genai).\n"
        )

class MessageHolder:
    """hold messages for model conversion and upload operations."""

    def __init__(self):
        self.str_messages = []
        self.dict_messages = {}

    def add(self, msg):
        if isinstance(msg, str):
            self.str_messages.append(msg)
        else:
            # msg: {
            #     f"{execution_provider}-{precision}-{name}": {
            #         "🔁 Conversion": "⏳",
            #         "📤 Upload": "⏳"
            #     }
            # }
            for name, value in msg.items():
                if name not in self.dict_messages:
                    self.dict_messages[name] = value
                self.dict_messages[name].update(value)
        return self

    def markdown(self):
        all_keys = list(dict.fromkeys(
            key for value in self.dict_messages.values() for key in value
        ))

        header = "| Name | " + " | ".join(all_keys) + " |"
        divider = "|------|" + "|".join(["------"] * len(all_keys)) + "|"
        rows = []
        for name, steps in self.dict_messages.items():
            row = [f"`{name}`"]
            for key in all_keys:
                row.append(steps.get(key, ""))
            rows.append("| " + " | ".join(row) + " |")

        lines = []
        for msg in self.str_messages:
            lines.append("")
            lines.append(msg)
        if rows:
            lines.append("")
            lines.append(header)
            lines.append(divider)
            lines.extend(rows)

        return "\n".join(lines)

class RedirectHandler(logging.Handler):
    """Handles logging redirection to progress logger."""

    def __init__(self, context: contextvars.ContextVar = None):
        super().__init__(logging.NOTSET)
        self.context = context

    def emit(self, record: logging.LogRecord):
        progress_logger = self.context.get(None)

        if progress_logger:
            try:
                progress_logger.logger.handle(record)
            except Exception as e:
                logging.getLogger(__name__).debug(f"Failed to forward log: {e}")
        else:
            logging.getLogger(__name__).handle(record)

if __name__ == "__main__":
    # context progress logger
    progress_logger_ctx = contextvars.ContextVar("progress_logger", default=None)

    # default config log_path
    config = Config.from_env()
    log_path = config.log_path / 'ui.log'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_path))
    logger.info("Gradio UI started")

with gr.Blocks() as demo:
    gr_user_config = gr.State(config)
    gr.Markdown("## 🤗 Convert HuggingFace Models to ONNX (ONNX Runtime GenAI Version)")
    gr_input_model_id = gr.Textbox(label="Model ID", info="e.g. microsoft/Phi-3-mini-4k-instruct")
    gr_user_token = gr.Textbox(label="HF Token (Optional)", type="password", visible=False)
    gr_same_repo = gr.Checkbox(label="Upload to same repo (if you own it)", visible=False, info="Do you want to upload the ONNX weights to the same repository?")
    gr_proceed = gr.Button("Convert and Upload", interactive=False)
    gr_result = gr.Markdown("")

    gr_input_model_id.change(
        fn=lambda x: [gr.update(visible=x != ""), gr.update(interactive=x != "")],
        inputs=[gr_input_model_id],
        outputs=[gr_user_token, gr_proceed],
        api_name=False
    )

    def change_user_token(input_model_id, user_hf_token, user_config):
        # update hf_token
        try:
            user_config.token(user_hf_token)
        except Exception as e:
            gr.Error(str(e), duration=5)
        if user_hf_token != "":
            if user_config.hf_username == input_model_id.split("/")[0]:
                return [gr.update(visible=True), user_config]
        return [gr.update(visible=False), user_config]
    gr_user_token.change(
        fn=change_user_token,
        inputs=[gr_input_model_id, gr_user_token, gr_user_config],
        outputs=[gr_same_repo, gr_user_config],
        api_name=False
    )

    def click_proceed(input_model_id, same_repo, user_config, progress=gr.Progress(track_tqdm=True)):
        converter = ModelConverter(user_config, progress_logger_ctx)
        holder = MessageHolder()

        input_model_id = input_model_id.strip()
        model_name = input_model_id.split("/")[-1]
        output_model_id = f"{user_config.hf_username}/{model_name}"

        if not same_repo:
            output_model_id += "-onnx-genai"
        if not same_repo and converter.api.repo_exists(output_model_id):
            yield gr.update(interactive=True), "This model has already been converted! 🎉"
            if user_config.ignore_converted:
                yield gr.update(interactive=True), "Ignore it, continue..."
            else:
                return

        # update markdown
        for task in converter.list_tasks():
            yield gr.update(interactive=False), holder.add(task).markdown()

        # update log
        logger = user_config.logger
        logger_path = user_config.logger_path
        logger.info(f"Log file: {logger_path}")
        yield gr.update(interactive=False), \
            holder.add(f"# 📄 Log file [{user_config.id}](./gradio_api/file={logger_path})").markdown()

        # update counter
        with suppress(Exception):
            requests.get("https://counterapi.com/api/xiaoyao9184.github.com/view/convert-to-genai")

        # update markdown
        logger.info("Conversion started...")
        gen = converter.convert_model(
            input_model_id, output_model_id, lambda n=-1: progress.update(n)
        )
        try:
            while True:
                msg = next(gen)
                yield gr.update(interactive=False), holder.add(msg).markdown()
        except StopIteration as e:
            output_dir = e.value
            yield gr.update(interactive=True), \
                holder.add(f"🔁 Conversion successful✅! 📁 output to {output_dir}").markdown()
        except Exception as e:
            logger.exception(e)
            yield gr.update(interactive=True), holder.add("🔁 Conversion failed🚫").markdown()
            return

        # update markdown
        logger.info("Upload started...")
        gen = converter.upload_model(input_model_id, output_model_id, lambda n=-1: progress.update(n))
        try:
            while True:
                msg = next(gen)
                yield gr.update(interactive=False), holder.add(msg).markdown()
        except StopIteration as e:
            output_model_url = f"{user_config.hf_base_url}/{output_model_id}"
            yield gr.update(interactive=True), \
                holder.add(f"📤 Upload successful✅! 📦 Go to [{output_model_id}]({output_model_url}/tree/main)").markdown()
        except Exception as e:
            logger.exception(e)
            yield gr.update(interactive=True), holder.add("📤 Upload failed🚫").markdown()
            return
    gr_proceed.click(
        fn=click_proceed,
        inputs=[gr_input_model_id, gr_same_repo, gr_user_config],
        outputs=[gr_proceed, gr_result]
    )

if __name__ == "__main__":
    # redirect builtins.print to context progress logger
    def context_aware_print(*args, **kwargs):
        progress_logger = progress_logger_ctx.get(None)
        if progress_logger:
            progress_logger.print(*args, **kwargs)
        else:
            builtins._original_print(*args, **kwargs)
    builtins._original_print = builtins.print
    builtins.print = context_aware_print

    # redirect sys.stdout.write to context progress logger
    def context_aware_write(text):
        progress_logger = progress_logger_ctx.get(None)
        if progress_logger:
            progress_logger.write(text.rstrip(), sys.stdout._original_write)
        else:
            sys.stdout._original_write(text)
    sys.stdout._original_write = sys.stdout.write
    sys.stdout.write = context_aware_write

    # redirect logger to context progress logger
    handler = RedirectHandler(progress_logger_ctx)
    for logger in [logging.getLogger("onnxruntime"), logging.getLogger("huggingface_hub.hf_api")]:
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logger.level)
        logger.propagate = False

    demo.launch(server_name="0.0.0.0", allowed_paths=[os.path.realpath(config.log_path.parent)])
