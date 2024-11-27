import chainlit as cl



from typing import Optional



import time







# Store conversation history



conversation_memory = []







@cl.on_chat_start



async def start():



    """Initializes the chat session"""



    # Send an initial message



    await cl.Message(



        content="👋 Hello! I'm your AI assistant. How can I help you today?",



        author="Assistant"



    ).send()



    



    # Set some session variables



    cl.user_session.set("conversation_started", True)







@cl.on_message



async def main(message: cl.Message):



    """Main message handler"""



    



    # Simulate some processing time



    with cl.Step("Processing...") as step:



        time.sleep(1)  # Simulated delay



        step.output = "Processed message"



    



    # Store message in conversation history



    conversation_memory.append({



        "role": "user",



        "content": message.content



    })



    



    # Create a response



    response = f"I received your message: '{message.content}'. This is a demo response."



    



    # Store response in conversation history



    conversation_memory.append({



        "role": "assistant",



        "content": response



    })



    



    # Send response with typing effect



    await cl.Message(



        content=response,



        author="Assistant"



    ).send()







@cl.password_auth_callback



def auth_callback(username: str, password: str) -> Optional[cl.User]:



    """Basic authentication handler"""



    # This is a simple example - in production, use proper authentication



    if username == "demo" and password == "password":



        return cl.User(identifier="demo", metadata={"role": "user"})



    return None







@cl.on_chat_end



async def end():



    """Cleanup when chat ends"""



    await cl.Message(content="👋 Thank you for chatting! Goodbye!").send()







# Custom action handler example



@cl.action_callback("feedback")



async def on_action(action):



    """Handles custom feedback action"""



    await cl.Message(content=f"Received feedback: {action.value}").send()
