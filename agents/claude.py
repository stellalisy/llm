import os
import logging
from types import SimpleNamespace

import boto3
import json
import time

from .base import AsyncBaseAgent
from pydantic import BaseModel
import yaml

logger = logging.getLogger(__name__)

class Recipe(BaseModel):
  recipe_name: str
  ingredients: list[str]

CLAUDE_KEYS_FILE = os.path.join(os.path.dirname(__file__), "..", "claude_keys_uptodate.txt")

CREDENTIAL_EXPIRED_MARKERS = [
    "security token included in the request is invalid",
    "security token included in the request is expired",
    "UnrecognizedClientException",
    "ExpiredTokenException",
    "InvalidIdentityToken",
    "The security token",
]

# Models that support structured output
STRUCTURED_OUTPUT_MODELS = ["claude-3-5-sonnet-20241022-v1:0", "claude-3-5-haiku-20241007-v1:0", "claude-3-5-sonnet-20250219-v1:0"]


class AsyncClaudeAgent(AsyncBaseAgent):
    CREDENTIAL_REFRESH_COOLDOWN = 60  # seconds between refresh attempts

    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self._last_refresh_time = 0
        self._last_keys_mtime = 0
        
        if not os.path.exists(self.args.api_info):
            raise ValueError(f"API info file {self.args.api_info} not found")
        with open(self.args.api_info, 'r') as f:
            self.api_info = yaml.safe_load(f).get(self.args.api_account, {})

        self._init_client_from_credentials()

    def _parse_claude_keys_file(self):
        """Read credentials from claude_keys_uptodate.txt."""
        if not os.path.exists(CLAUDE_KEYS_FILE):
            return None
        creds = {}
        with open(CLAUDE_KEYS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line:
                    key, val = line.split("=", 1)
                    creds[key.strip()] = val.strip()
        if "AWS_ACCESS_KEY_ID" in creds:
            return creds
        return None

    def _get_credentials(self):
        """Get credentials, preferring claude_keys_uptodate.txt over api_info.yaml."""
        file_creds = self._parse_claude_keys_file()
        if file_creds:
            return {
                "aws_access_key_id": file_creds["AWS_ACCESS_KEY_ID"],
                "aws_secret_access_key": file_creds["AWS_SECRET_ACCESS_KEY"],
                "aws_session_token": file_creds.get("AWS_SESSION_TOKEN", ""),
                "expiry_time": file_creds.get("EXPIRY_TIME", "unknown"),
            }
        return {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", self.api_info.get("aws_access_key_id", "")),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", self.api_info.get("aws_secret_access_key", "")),
            "aws_session_token": os.environ.get("AWS_SESSION_TOKEN", self.api_info.get("aws_session_token", "")),
        }

    def _init_client_from_credentials(self):
        """(Re)initialize the boto3 Bedrock client with current credentials."""
        from botocore.config import Config
        creds = self._get_credentials()
        config = Config(read_timeout=1000)
        self.client = boto3.client(
            "bedrock-runtime",
            region_name="us-west-2",
            aws_access_key_id=creds["aws_access_key_id"],
            aws_secret_access_key=creds["aws_secret_access_key"],
            aws_session_token=creds["aws_session_token"],
            config=config,
        )
        if os.path.exists(CLAUDE_KEYS_FILE):
            self._last_keys_mtime = os.path.getmtime(CLAUDE_KEYS_FILE)
        expiry = creds.get("expiry_time", "unknown")
        logger.info(f"Claude client initialized (key_id=...{creds['aws_access_key_id'][-4:]}, expires={expiry})")

    def _is_credential_error(self, error: Exception) -> bool:
        err_str = str(error)
        return any(marker.lower() in err_str.lower() for marker in CREDENTIAL_EXPIRED_MARKERS)

    def _should_attempt_refresh(self) -> bool:
        """Check if enough time has passed and the keys file has been updated."""
        now = time.time()
        if now - self._last_refresh_time < self.CREDENTIAL_REFRESH_COOLDOWN:
            return False
        if os.path.exists(CLAUDE_KEYS_FILE):
            current_mtime = os.path.getmtime(CLAUDE_KEYS_FILE)
            if current_mtime > self._last_keys_mtime:
                return True
        return now - self._last_refresh_time >= self.CREDENTIAL_REFRESH_COOLDOWN

    def _refresh_credentials_and_retry(self) -> bool:
        """Attempt to refresh credentials from file and reinitialize client.
        Returns True if credentials were actually updated, False if file unchanged."""
        if not self._should_attempt_refresh():
            return False
        self._last_refresh_time = time.time()
        if os.path.exists(CLAUDE_KEYS_FILE):
            current_mtime = os.path.getmtime(CLAUDE_KEYS_FILE)
            if current_mtime > self._last_keys_mtime:
                logger.warning("Keys file updated, refreshing credentials from claude_keys_uptodate.txt...")
                self._init_client_from_credentials()
                return True
            else:
                logger.warning(
                    f"Credential error but keys file unchanged (mtime={current_mtime}). "
                    f"Waiting for external refresh of {CLAUDE_KEYS_FILE}"
                )
                return False
        logger.warning(f"Credential error but {CLAUDE_KEYS_FILE} not found.")
        return False
    
    def interact(self, prompt, temperature=0, max_tokens=256, history=None, json_mode=False, response_format=None, **kwargs):
        if json_mode:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, json_mode=True, temperature=temperature, max_tokens=max_tokens)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, json_mode=True, temperature=temperature, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")
            
        elif response_format and self.args.model in STRUCTURED_OUTPUT_MODELS:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, response_format=response_format, temperature=temperature, max_tokens=max_tokens)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, response_format=response_format, temperature=temperature, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")
            
        else:
            if isinstance(prompt, str):
                outputs = self._generate(prompt, temperature=temperature, max_tokens=max_tokens, history=history)
            elif isinstance(prompt, list):
                outputs = self._generate_from_messages(prompt, temperature=temperature, max_tokens=max_tokens)
            else:
                raise ValueError("Prompt must be a string or a list of dictionaries")
            
        responses = self._postprocess_output(outputs, response_format=response_format)

        return responses

    def _generate(self, prompt, json_mode=False, temperature=None, max_tokens=None, history=None, response_format=None):
        # Prepare messages
        
        retries = 5
        while retries > 0:
            try:
                messages = []
                # Add history if provided
                if history:
                    messages.extend(history[len(history)-3+retries:])
                
                # Add current prompt
                if isinstance(prompt, str):
                    messages.append({"role": "user", "content": prompt})
                else:
                    messages.extend(prompt)
                
                # Prepare body for Bedrock API
                body = {
                    "messages": messages,
                    "anthropic_version": "bedrock-2023-05-31"
                }
                
                # Add parameters
                if max_tokens is not None:
                    body["max_tokens"] = max_tokens
                elif hasattr(self.args, 'max_tokens'):
                    body["max_tokens"] = self.args.max_tokens
                    
                if temperature is not None:
                    body["temperature"] = temperature
                elif hasattr(self.args, 'temperature'):
                    body["temperature"] = self.args.temperature
                
                # Handle JSON mode / structured output (only for supported models)
                if json_mode and self.args.model in STRUCTURED_OUTPUT_MODELS:
                    body["response_format"] = {"type": "json_object"}
                
                elif response_format and self.args.model in STRUCTURED_OUTPUT_MODELS:
                    body["response_format"] = response_format
                    # Note: Claude's structured output is handled differently than Gemini
                    # You might need to add the schema to the prompt or use a different approach
                
                # Generate content using Bedrock
                response = self.client.invoke_model(
                    body=json.dumps(body),
                    modelId=f"arn:aws:bedrock:us-west-2:396608793503:inference-profile/us.{self.args.model}"
                )
                return response
            except Exception as e:
                if self._is_credential_error(e):
                    refreshed = self._refresh_credentials_and_retry()
                    if refreshed:
                        continue
                    # File not updated yet -- wait and let external process refresh it
                    logger.warning(f"Credentials expired, waiting {self.CREDENTIAL_REFRESH_COOLDOWN}s for external key refresh...")
                    time.sleep(self.CREDENTIAL_REFRESH_COOLDOWN)
                    retries -= 1
                    if retries == 0:
                        raise Exception(f"Failed to generate response: {e}")
                    continue
                err_msg = str(e).lower()
                if any(p in err_msg for p in ("too long", "exceed", "token limit", "too large", "payload size")):
                    prompt = prompt[int(len(prompt) * 0.4):]
                elif "please wait" in str(e):
                    time.sleep(20)
                retries -= 1
                if retries == 0:
                    raise Exception(f"Failed to generate response: {e}")
        

    def _generate_from_messages(self, messages, json_mode=False, temperature=None, max_tokens=None, response_format=None):
        # Messages are already in the correct format for Claude
        return self._generate(messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens, response_format=response_format)

    def preprocess_input(self, text):
        return text

    def _postprocess_output(self, output, response_format=None):
        response_body = json.loads(output.get("body").read())
        content = response_body.get("content", [])
        
        # Extract the text from the content array
        response_text = ""
        if content and len(content) > 0:
            response_text = content[0].get("text", "")
        
        # Extract usage information
        usage = response_body.get("usage", {})
        
        return {
            "response_text": response_text,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "finish_reason": response_body.get("stop_reason", None),
            "cached_tokens": usage.get("cached_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
        }