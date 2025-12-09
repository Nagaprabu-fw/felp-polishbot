import hashlib
import hmac
import time
import json
import os
import logging
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import Response
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate




# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('/tmp/app.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# load_dotenv()

app = FastAPI()

# # Configuration
# SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
# SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# # Initialize Slack client
# logger.info("Initializing Slack client")
# slack_client = WebClient(token=SLACK_BOT_TOKEN)

# # Initialize Azure OpenAI with LangChain
# logger.info(f"Initializing Azure OpenAI with endpoint: {AZURE_OPENAI_ENDPOINT}")
# llm = AzureChatOpenAI(
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_API_KEY,
#     api_version=AZURE_OPENAI_API_VERSION,
#     deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
#     temperature=0.7,
# )
# logger.info("Azure OpenAI initialized successfully")

# # Tone definitions
# TONES = {
#     "professional": {
#         "name": "Professional",
#         "description": "Formal and business-appropriate",
#         "prompt": "Rewrite the following text in a professional and formal business tone. Maintain the core message but make it suitable for corporate communication:"
#     },
#     "friendly": {
#         "name": "Friendly",
#         "description": "Warm and approachable",
#         "prompt": "Rewrite the following text in a friendly and warm tone. Make it conversational and approachable while keeping the main message:"
#     },
#     "concise": {
#         "name": "Concise",
#         "description": "Brief and to the point",
#         "prompt": "Rewrite the following text to be concise and brief. Remove unnecessary words while preserving the essential meaning:"
#     },
#     "enthusiastic": {
#         "name": "Enthusiastic",
#         "description": "Energetic and positive",
#         "prompt": "Rewrite the following text in an enthusiastic and energetic tone. Make it positive and engaging while maintaining the core message:"
#     }
# }


# def verify_slack_request(request_body: bytes, timestamp: str, signature: str) -> bool:
#     """Verify that the request came from Slack by validating the signature."""
#     if not SLACK_SIGNING_SECRET:
#         logger.error("SLACK_SIGNING_SECRET not set")
#         raise ValueError("SLACK_SIGNING_SECRET not set")
    
#     # Prevent replay attacks - reject requests older than 5 minutes
#     if abs(time.time() - int(timestamp)) > 60 * 5:
#         logger.warning(f"Request timestamp too old: {timestamp}")
#         return False
    
#     # Create the signature base string
#     sig_basestring = f"v0:{timestamp}:{request_body.decode('utf-8')}"
    
#     # Create the signature hash
#     my_signature = 'v0=' + hmac.new(
#         SLACK_SIGNING_SECRET.encode(),
#         sig_basestring.encode(),
#         hashlib.sha256
#     ).hexdigest()
    
#     # Compare signatures using constant-time comparison
#     return hmac.compare_digest(my_signature, signature)


# def create_tone_selection_blocks(text: str) -> list:
#     """Create Slack blocks with tone selection buttons."""
#     return [
#         {
#             "type": "section",
#             "text": {
#                 "type": "mrkdwn",
#                 "text": f"*Text to polish:*\n> {text}\n\nSelect a tone:"
#             }
#         },
#         {
#             "type": "actions",
#             "block_id": "tone_selection",
#             "elements": [
#                 {
#                     "type": "button",
#                     "text": {
#                         "type": "plain_text",
#                         "text": f"✨ {TONES[tone]['name']}"
#                     },
#                     "value": json.dumps({"tone": tone, "text": text}),
#                     "action_id": f"polish_{tone}"
#                 }
#                 for tone in TONES.keys()
#             ]
#         }
#     ]


# async def polish_text_with_llm(text: str, tone: str) -> str:
#     """Use Azure OpenAI to polish the text with the selected tone."""
#     logger.info(f"Polishing text with tone: {tone}")
#     tone_config = TONES.get(tone)
#     if not tone_config:
#         logger.error(f"Invalid tone selected: {tone}")
#         return "Invalid tone selected."
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a helpful writing assistant that rewrites text in different tones."),
#         ("user", f"{tone_config['prompt']}\n\nText: {text}")
#     ])
    
#     try:
#         chain = prompt | llm
#         logger.debug(f"Sending request to Azure OpenAI for text length: {len(text)}")
#         response = await chain.ainvoke({})
#         logger.info("Successfully received response from Azure OpenAI")
#         return response.content
#     except Exception as e:
#         logger.error(f"Error polishing text with LLM: {str(e)}", exc_info=True)
#         raise


# def send_polished_response(channel_id: str, user_id: str, original_text: str, polished_text: str, tone: str):
#     """Send the polished text back to the user."""
#     try:
#         logger.info(f"Sending polished response to user {user_id} in channel {channel_id}")
        
#         blocks = [
#             {
#                 "type": "section",
#                 "text": {
#                     "type": "mrkdwn",
#                     "text": f"*Original:*\n> {original_text}"
#                 }
#             },
#             {
#                 "type": "divider"
#             },
#             {
#                 "type": "section",
#                 "text": {
#                     "type": "mrkdwn",
#                     "text": f"*Polished ({TONES[tone]['name']} tone):*\n{polished_text}"
#                 }
#             }
#         ]
        
#         # Check if the channel is a DM (starts with 'D') or a channel/group
#         is_dm = channel_id.startswith('D')
#         logger.info(f"Channel type: {'DM' if is_dm else 'Channel/Group'} (ID: {channel_id})")
        
#         if is_dm:
#             # For DMs, we need to open a conversation first
#             try:
#                 logger.info(f"Opening DM conversation with user {user_id}")
#                 dm_response = slack_client.conversations_open(users=user_id)
#                 dm_channel_id = dm_response["channel"]["id"]
#                 logger.info(f"DM channel opened: {dm_channel_id}")
                
#                 slack_client.chat_postMessage(
#                     channel=dm_channel_id,
#                     text=f"Polished text ({TONES[tone]['name']} tone)",
#                     blocks=blocks
#                 )
#             except SlackApiError as dm_error:
#                 logger.error(f"Error with DM, falling back to ephemeral: {dm_error.response['error']}")
#                 # Fallback to ephemeral if DM fails
#                 slack_client.chat_postEphemeral(
#                     channel=channel_id,
#                     user=user_id,
#                     text=f"Polished text ({TONES[tone]['name']} tone)",
#                     blocks=blocks
#                 )
#         else:
#             # For channels/groups, use chat.postEphemeral
#             slack_client.chat_postEphemeral(
#                 channel=channel_id,
#                 user=user_id,
#                 text=f"Polished text ({TONES[tone]['name']} tone)",
#                 blocks=blocks
#             )
        
#         logger.info("Successfully sent polished response to Slack")
#     except SlackApiError as e:
#         logger.error(f"Error sending message to Slack: {e.response['error']}", exc_info=True)
#         # Try one last fallback - post to user as DM
#         try:
#             logger.info("Attempting final fallback to user DM")
#             dm_response = slack_client.conversations_open(users=user_id)
#             dm_channel_id = dm_response["channel"]["id"]
#             slack_client.chat_postMessage(
#                 channel=dm_channel_id,
#                 text=f"Polished text ({TONES[tone]['name']} tone)\n\n*Original:* {original_text}\n\n*Polished:* {polished_text}"
#             )
#         except Exception as fallback_error:
#             logger.error(f"All message sending attempts failed: {fallback_error}", exc_info=True)


# @app.post("/slack/polish")
# async def slack_polish_command(request: Request):
#     """Handle /polish slash command."""
#     logger.info("Received /polish command")
#     # Get request headers and body
#     timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
#     signature = request.headers.get("X-Slack-Signature", "")
#     body = await request.body()
    
#     # Verify the request is from Slack
#     if not verify_slack_request(body, timestamp, signature):
#         logger.warning("Invalid Slack request signature for /polish command")
#         raise HTTPException(status_code=401, detail="Invalid request signature")
    
#     # Parse form data
#     form_data = await request.form()
#     text = form_data.get("text", "").strip()
#     logger.info(f"Processing polish command with text length: {len(text)}")
    
#     # Check if text is provided
#     if not text:
#         logger.warning("Empty text provided for polish command")
#         return Response(
#             content=json.dumps({
#                 "response_type": "ephemeral",
#                 "text": "Please provide text to polish. Usage: `/polish Your text here`"
#             }),
#             media_type="application/json",
#             status_code=200
#         )
    
#     # Return tone selection buttons
#     response_data = {
#         "response_type": "ephemeral",
#         "blocks": create_tone_selection_blocks(text)
#     }
    
#     return Response(
#         content=json.dumps(response_data),
#         media_type="application/json",
#         status_code=200
#     )


# @app.post("/slack/interactions")
# async def slack_interactions(request: Request, background_tasks: BackgroundTasks):
#     """Handle Slack interactive components (button clicks)."""
#     logger.info("Received interaction request")
#     # Get request headers and body
#     timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
#     signature = request.headers.get("X-Slack-Signature", "")
#     body = await request.body()
    
#     # Verify the request is from Slack
#     if not verify_slack_request(body, timestamp, signature):
#         logger.warning("Invalid Slack request signature for interaction")
#         raise HTTPException(status_code=401, detail="Invalid request signature")
    
#     # Parse the payload
#     form_data = await request.form()
#     payload = json.loads(form_data.get("payload", "{}"))
    
#     # Extract action information
#     action = payload.get("actions", [{}])[0]
#     action_id = action.get("action_id", "")
    
#     if action_id.startswith("polish_"):
#         # Extract tone and text from button value
#         value_data = json.loads(action.get("value", "{}"))
#         tone = value_data.get("tone")
#         text = value_data.get("text")
        
#         user_id = payload["user"]["id"]
#         channel_id = payload["channel"]["id"]
#         logger.info(f"Processing polish action with tone: {tone} for user: {user_id}")
        
#         # Acknowledge the interaction immediately
#         response_data = {
#             "response_type": "ephemeral",
#             "replace_original": True,
#             "text": f"✨ Polishing your text with {TONES[tone]['name']} tone... Please wait."
#         }
        
#         # Process in background
#         async def process_polish():
#             polished_text = await polish_text_with_llm(text, tone)
#             send_polished_response(channel_id, user_id, text, polished_text, tone)
        
#         background_tasks.add_task(process_polish)
        
#         return Response(
#             content=json.dumps(response_data),
#             media_type="application/json",
#             status_code=200
#         )
    
#     return Response(content=json.dumps({}), media_type="application/json")


# @app.post("/slack/hello")
# async def slack_hello_command(request: Request):
#     """Handle /hello slash command (for testing)."""
#     timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
#     signature = request.headers.get("X-Slack-Signature", "")
#     body = await request.body()
    
#     if not verify_slack_request(body, timestamp, signature):
#         raise HTTPException(status_code=401, detail="Invalid request signature")
    
#     response_data = {
#         "response_type": "ephemeral",
#         "text": "Hello World! 👋"
#     }
    
#     return Response(
#         content=json.dumps(response_data),
#         media_type="application/json",
#         status_code=200
#     )


@app.get("/")
async def root():
    return {"message": "Hello, Welcome to polishbot homepage"}


@app.get("/health")
async def health_check():
    """Health check endpoint for ALB target group."""
    logger.debug("Health check requested")
    return {"status": "healthy"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
