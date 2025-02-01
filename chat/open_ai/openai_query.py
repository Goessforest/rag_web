from openai.types.chat.chat_completion import ChatCompletion
from openai import OpenAI
import os
from typing import *
import inspect
from functools import partial

import json

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_INSTRUCTION = "You are a helpfull assistant."

class AI_Message:
    def __init__(self, role: Literal['user', 'system'] = "user", 
                 content: str = "", 
                 function_call: str = None,
                  refusal: bool = None,
                  tool_calls: str = None):
        self.role = role
        self.content = content
        self.function_call = function_call
        self.refusal = refusal
        self.tool_calls = tool_calls

        """
        ChatCompletion(id='chatcmpl-AvX1G5HGhF8CMlXWAQjiPgvtW6OqD', 
               choices=[
                   Choice(finish_reason='tool_calls', 
                          index=0, 
                          logprobs=None, 
                          message=
                            ChatCompletionMessage(content=None, 
                                                refusal=None, 
                                                role='assistant', 
                                                audio=None, 
                                                function_call=None, 
                                                tool_calls=[
                                                    ChatCompletionMessageToolCall(id='call_8GKM0d44pTeuhZl2AUteFZEI', 
                                                                                  function=Function(arguments='{\n  "a": 1,\n  "b": "4"\n}', 
                                                                                                    name='test_func'), 
                                                                                  type='function')]))], 
                                                                                  created=1738274986, 
                                                                                  model='gpt-4-0613', 
                                                                                  object='chat.completion', 
                                                                                  service_tier='default', 
                                                                                  system_fingerprint=None, 
                                                                                  usage=CompletionUsage(completion_tokens=23, 
                                                                                                        prompt_tokens=71, 
                                                                                                        total_tokens=94, 
                                                                                                        completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, 
                                                                                                                                                          audio_tokens=0, 
                                                                                                                                                          reasoning_tokens=0, 
                                                                                                                                                          rejected_prediction_tokens=0), 
                                                                                                        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))
    """

    def __repr__(self):
        return f"AI_Message(role={self.role}, content={self.content}, function_call={self.function_call}, refusal={self.refusal}, tool_calls={self.tool_calls})"
    
    def __str__(self):
        if self.content is None:
            return str()
        return f"{self.content}"

    def to_dict(self):
        return {"role": self.role, "content": self.content, "function_call": self.function_call, "refusal": self.refusal, "tool_calls": self.tool_calls}
    




class OpenAIQuery:
    def __init__(self):
        self.openai_clinet = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        )
        self.messages = []
    

    def parse_function_to_tool(self, func):
        """
        Parses a Python function into the specified "tool" format.

        Args:
            func (function): The function to parse.

        Returns:
            dict: A dictionary in the "tool" format.
        """
        # Get function name, docstring, and signature
        func_name = func.__name__
        docstring = func.__doc__ or "No description provided."
        signature = inspect.signature(func)

        # Build the parameters schema
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        for param_name, param in signature.parameters.items():
            # Define parameter schema
            param_schema = {}
            
            # Add type if available
            if param.annotation != inspect.Parameter.empty:
                python_type = param.annotation
                if python_type == int:
                    param_schema["type"] = "integer"
                elif python_type == float:
                    param_schema["type"] = "number"
                elif python_type == str:
                    param_schema["type"] = "string"
                elif python_type == bool:
                    param_schema["type"] = "boolean"
                elif python_type == dict:
                    param_schema["type"] = "object"
                elif python_type == list:
                    param_schema["type"] = "array"
                else:
                    param_schema["type"] = "string"  # Fallback for unsupported types

            # Add default value if present
            if param.default != inspect.Parameter.empty:
                param_schema["default"] = param.default
            else:
                # Mark as required if no default value
                parameters["required"].append(param_name)

            # Add parameter schema to properties
            parameters["properties"][param_name] = param_schema

        # Build the tool dictionary
        tool = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": docstring.strip(),
                "parameters": parameters,
                "strict": True,
            },
        }

        return tool


    def _ask_openai(self, messages: List[AI_Message], tools: List[Callable]=[], max_tokens:int=100, temperature:float=DEFAULT_TEMPERATURE, ai_model:str=DEFAULT_MODEL) -> ChatCompletion:
        """Sends the data to openai and returns the response"""

        # Parse messages
        if len(messages) == 0:
            raise ValueError("messages must not be at least one message")
        elif len(messages) == 1:
            messages.insert(0, AI_Message(role="system", content=DEFAULT_INSTRUCTION))
        
        self.messages = messages
        list_of_messages = [message.to_dict() for message in messages]

        # add tools
        tools = [self.parse_function_to_tool(tool) for tool in tools] or None
        response = self.openai_clinet.chat.completions.create(
                messages=list_of_messages,
                tools=tools,
                model=ai_model,
                max_tokens=max_tokens,
                temperature=temperature
            )
        choice = response.choices[0].message
        if choice.tool_calls:
            try:
                bundle = AI_Message(role=choice.role, content=choice.content, function_call=choice.tool_calls[0].function, refusal=choice.refusal, tool_calls=choice.tool_calls)
            except AttributeError:
                bundle = AI_Message(role=choice.role, content=choice.content, function_call=None, refusal=choice.refusal, tool_calls=choice.tool_calls)

        else:
            bundle = AI_Message(role=choice.role, content=choice.content, function_call=None, refusal=choice.refusal, tool_calls=choice.tool_calls)

        self.messages.append(bundle)
        return bundle
    
    def execute_tool_if_possible(self, response:AI_Message, funcs:Union[List[Callable], Callable]=[]):
        """Execute a tool function"""
        
        if funcs and response.function_call:
            name = response.function_call.name
            args = json.loads(response.function_call.arguments)
            funcs = funcs if isinstance(funcs, list) else [funcs]
            for func in funcs:
                if func.__name__ == name and callable(func):
                    return func(**args)

        return response
    

    def query(self, prompt:str, 
              instructions:str=DEFAULT_INSTRUCTION, 
              messages:List[AI_Message]=[],
              func:Union[List[Callable], Callable]=[], 
              max_tokens:int=100, 
              temperature:float=DEFAULT_TEMPERATURE, 
              ai_model:str=DEFAULT_MODEL) -> AI_Message:
        """Ask a question and get a response from Openai based on the query"""
        if not messages:
            messages.append(AI_Message(role="system", content=instructions))
            messages.append(AI_Message(role="user", content=prompt))
        tools = func if isinstance(func, list) else [func]

        response = self._ask_openai(messages=messages, 
                         tools=tools,
                         max_tokens=max_tokens,
                         temperature=temperature,
                         ai_model=ai_model)
        # reset messages, due to unexpected behavior
        messages = []

        return self.execute_tool_if_possible(response, func)
    

    def follow_up_query(self, prompt:str, 
                        instructions:str=DEFAULT_INSTRUCTION,
                        messages:List[AI_Message]=[],
              func:Union[List[Callable], Callable]=[], 
              max_tokens:int=100, 
              temperature:float=DEFAULT_TEMPERATURE, 
              ai_model:str=DEFAULT_MODEL) -> str:
        """Ask a question and get a response from Openai based on the query"""
        if not messages:
            if not self.messages:
                self.messages.append(AI_Message(role="user", content=instructions))
            self.messages.append(AI_Message(role="user", content=prompt))
        else:
            self.messages.extend(messages)
        tools = func if isinstance(func, list) else [func]

        response = self._ask_openai(messages=self.messages, 
                         tools=tools,
                         max_tokens=max_tokens,
                         temperature=temperature,
                         ai_model=ai_model)
        return self.execute_tool_if_possible(response, func)
    
        


        