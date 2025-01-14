from guidance import models, gen
from guidance import user, system, assistant

import litellm
litellm.api_base = "http://bob.snu.vision:11434"

from third_party.guidance.custom_completion import CustomLiteLLMCompletion, image

if __name__ == "__main__":
    llava_34b = CustomLiteLLMCompletion(
        model="ollama/llava:34b",
    )

    with user():
        llava_34b = llava_34b + "What is this a picture of?" + image("longs_peak.jpg")

    with assistant():
        llava_34b += gen("answer")
        
    print(llava_34b)