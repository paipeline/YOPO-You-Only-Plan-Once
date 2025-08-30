from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import logging
import time
from colorama import init, Fore, Back, Style
import os
load_dotenv()
init(autoreset=True)  # Initialize colorama

# Configure colorful logging
class ColorfulFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Back.WHITE
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        record.msg = f"{log_color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# Set up colorful logging
handler = logging.StreamHandler()
handler.setFormatter(ColorfulFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to DEBUG to see more details, INFO for less
logger.handlers = [handler]


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs):
        pass


class LangChainProvider(LLMProvider):
    def __init__(self, model_name: str, system_prompt: Optional[str] = None, **kwargs):
        if not model_name:
            raise ValueError("model_name cannot be None or empty")
        self.model_name = model_name
        self.system_prompt = system_prompt
        logger.info(f"Initializing LangChainProvider with model: {Fore.CYAN}{model_name}{Style.RESET_ALL}")
        if system_prompt:
            logger.info(f"Using custom system prompt: {Fore.YELLOW}{len(system_prompt)}{Style.RESET_ALL} chars")
        self.llm = self._initialize_model(model_name, **kwargs)
        logger.info(f"Successfully initialized {Fore.CYAN}{model_name}{Style.RESET_ALL}")
    
    def _initialize_model(self, model_name: str, **kwargs) -> BaseChatModel:
        if not model_name:
            raise ValueError("model_name cannot be None or empty")
        
        try:
            if model_name.startswith("gpt"):
                logger.debug(f"Creating ChatOpenAI instance for {model_name}")
                return ChatOpenAI(model=model_name, **kwargs)
            elif model_name.startswith("claude"):
                logger.debug(f"Creating ChatAnthropic instance for {model_name}")
                return ChatAnthropic(model=model_name, **kwargs)
            elif model_name.startswith("gemini"):
                logger.debug(f"Creating ChatGoogleGenerativeAI instance for {model_name}")
                return ChatGoogleGenerativeAI(model=model_name, **kwargs)
            else:
                logger.error(f"Unsupported model: {model_name}")
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {str(e)}")
            raise
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown message role: {role}")
        
        return langchain_messages
    
    def _add_system_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Add system prompt to messages if configured and not already present.
        
        Args:
            messages: Original messages list
            
        Returns:
            Messages list with system prompt prepended if needed
        """
        if not self.system_prompt:
            return messages
            
        # Check if first message is already a system message
        if messages and messages[0].get("role") == "system":
            # Override existing system message if custom prompt is set
            logger.debug("Overriding existing system message with custom prompt")
            return [{"role": "system", "content": self.system_prompt}] + messages[1:]
        else:
            # Prepend system prompt
            logger.debug("Adding custom system prompt to messages")
            return [{"role": "system", "content": self.system_prompt}] + messages
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        logger.info(f"Generating response with {Fore.CYAN}{self.model_name}{Style.RESET_ALL}, {Fore.YELLOW}{len(messages)}{Style.RESET_ALL} messages")
        start_time = time.time()
        
        try:
            # Prepend system prompt if configured and not already present
            messages_with_system = self._add_system_prompt(messages)
            langchain_messages = self._convert_messages(messages_with_system)
            response = self.llm.invoke(langchain_messages, **kwargs)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Generated response in {Fore.MAGENTA}{elapsed_time:.2f}s{Style.RESET_ALL}, length: {Fore.YELLOW}{len(response.content)}{Style.RESET_ALL} chars")
            
            # Only show preview for long responses
            if len(response.content) > 200:
                logger.debug(f"Response preview: {response.content[:100]}...")
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Response: {response.content}")
            
            return response.content
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs):
        logger.info(f"Starting stream generation with {Fore.CYAN}{self.model_name}{Style.RESET_ALL}, {Fore.YELLOW}{len(messages)}{Style.RESET_ALL} messages")
        start_time = time.time()
        char_count = 0
        
        try:
            # Prepend system prompt if configured and not already present
            messages_with_system = self._add_system_prompt(messages)
            langchain_messages = self._convert_messages(messages_with_system)
            for chunk in self.llm.stream(langchain_messages, **kwargs):
                char_count += len(chunk.content)
                yield chunk.content
            
            elapsed_time = time.time() - start_time
            logger.info(f"Stream completed in {Fore.MAGENTA}{elapsed_time:.2f}s{Style.RESET_ALL}, total chars: {Fore.YELLOW}{char_count}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Stream generation failed: {str(e)}")
            raise


class LLMProviderFactory:
    @staticmethod
    def create_provider(provider_type: str, model_name: str, system_prompt: Optional[str] = None, **kwargs) -> LLMProvider:
        logger.info(f"Creating provider: type={Fore.BLUE}{provider_type}{Style.RESET_ALL}, model={Fore.CYAN}{model_name}{Style.RESET_ALL}")
        
        if provider_type == "langchain":
            return LangChainProvider(model_name, system_prompt=system_prompt, **kwargs)
        else:
            logger.error(f"Unknown provider type: {provider_type}")
            raise ValueError(f"Unknown provider type: {provider_type}")

def llm_instance(model_name: str, system_prompt: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Get a LLM instance.
    
    Args:
        model_name: Name of the model to use
        system_prompt: Optional custom system prompt to prepend to all messages
        **kwargs: Additional arguments passed to the model
        
    Returns:
        LLMProvider instance
    """
    return LLMProviderFactory.create_provider("langchain", model_name, system_prompt=system_prompt, **kwargs)


if __name__ == "__main__":
    # Get model name from environment variable
    model_name = os.getenv("DEV_MODEL_NAME") or os.getenv("DEV_LLM_NAME") or "gpt-3.5-turbo"
    print(f"Using model: {model_name}")
    
    # Example 1: Without custom system prompt
    provider = llm_instance(model_name)
    print("\nExample 1 - Default behavior:")
    print(provider.generate([{"role": "user", "content": "Hello, how are you?"}]))
    print("\n" + "="*50 + "\n")
    
    # Example 2: With custom system prompt
    system_prompt = "You are a tag identifier agent. You are given a query and you need to identify the tags that are most relevant to the query."
    provider_with_prompt = llm_instance(model_name, system_prompt=system_prompt)
    print(provider_with_prompt.generate([{"role": "user", "content": "I want to learn about the history of the internet."}]))
    
    # Example 3: Stream with custom system prompt
    # print("\n" + "="*50 + "\n")
    # print("Example 3 - Stream with custom system prompt:")
    # for chunk in provider_with_prompt.generate_stream([{"role": "user", "content": "Tell me a joke"}]):
    #     print(chunk, end="", flush=True)
    # print()
