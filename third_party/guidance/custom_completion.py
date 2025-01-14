import re
import base64

from guidance.models._lite_llm import LiteLLMEngine
from guidance.models._grammarless import Grammarless
from guidance import guidance

_VARIABLES = dict()

class CustomLiteLLMCompletionEngine(LiteLLMEngine):
    def _prompt_to_messages(self, prompt):
        prompt = prompt.decode("utf8").split("\n")
        
        messages = []
        for i in range(0, len(prompt), 2):
            role = prompt[i].split("<|im_start|>")[-1]
            content = prompt[i+1].split("<|im_end|>")[0]
                
            new_content = dict()
            new_content['role'] = role
            new_content['content'] = []
            
            image_pattern = re.compile(r"<\|_image:(\d+)\|>")
            
            if image_pattern.search(content):
                image_id = image_pattern.search(content).group(1)
                new_content['content'].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(_VARIABLES[image_id]).decode('utf-8')}"}
                })
            new_content['content'].append({
                "type": "text",
                "text": image_pattern.sub("", content)
            })
                        
            messages.append(new_content)

        return messages[:-1]
    
    def _generator(self, prompt, temperature):

        # update our shared data state
        self._reset_shared_data(prompt, temperature)
        messages = self._prompt_to_messages(prompt)

        try:
            generator = self.litellm.completion(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=1,
                temperature=temperature,
                stream=True,
            )
        except Exception as e:  # TODO: add retry logic
            raise e

        for part in generator:
            chunk = part.choices[0].delta.content or ""
            yield chunk.encode("utf8")

class CustomLiteLLMCompletion(Grammarless):
    def __init__(
        self,
        model,
        tokenizer=None,
        echo=True,
        timeout=0.5,
        max_streaming_tokens=1000,
        compute_log_probs=False,
    ):
        super().__init__(
            CustomLiteLLMCompletionEngine(
                model,
                tokenizer,
                timeout,
                compute_log_probs,
                max_streaming_tokens,
            ),
            echo=echo,
        )
        
import http
import pathlib
import re
import typing
import urllib

@guidance
def image(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):

    # load the image bytes
    # ...from a url
    if isinstance(src, str) and re.match(r"[^:/]+://", src):
        with urllib.request.urlopen(src) as response:
            response = typing.cast(http.client.HTTPResponse, response)
            bytes_data = response.read()

    # ...from a local path
    elif allow_local and (isinstance(src, str) or isinstance(src, pathlib.Path)):
        with open(src, "rb") as f:
            bytes_data = f.read()

    # ...from image file bytes
    elif isinstance(src, bytes):
        bytes_data = src

    else:
        raise Exception(f"Unable to load image bytes from {src}!")

    bytes_id = str(id(bytes_data))

    # set the image bytes
    lm = lm.set(bytes_id, bytes_data)
    _VARIABLES[bytes_id] = bytes_data
    lm += f"<|_image:{bytes_id}|>"
    
    return lm
