import discord
import requests
import os
import time
import asyncio
import random
from uuid import uuid4
from dotenv import load_dotenv
from save_message import save_message_to_vector_db
from graph import generate_response, generate_intro

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)
last_message_time = time.time()
thread_id = ""

# Event listener for new messages
@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

    channel = client.get_channel(int(CHANNEL_ID))

    if channel:
        client.loop.create_task(periodic_messages(channel))
    else:
        print("Channel not found. Please check the channel ID.")

@client.event
async def on_message(message):
    global last_message_time
    global thread_id
    # Ignore messages from the bot itself
    if message.author == client.user:
        return
    
    try:
        question = f"[Author: {message.author.name} at {message.created_at.isoformat()}] Message: {message.content}"
        if time.time() - last_message_time <= 5 * 60:
            chat_id = thread_id

            response = await generate_response(question, chat_id)
            if len(response) > 0:
                await message.reply(response)
            last_message_time = time.time()
        elif client.user in message.mentions or client.user.name in message.content.lower():
            chat_id = uuid4()

            response = await generate_response(question, chat_id)
            if len(response) > 0:
                await message.reply(response)
            thread_id = chat_id
            last_message_time = time.time()

    except requests.exceptions.RequestException as e:
        print(f"Failed to respond message: {e}")
        raise

    # Save data
    try:
        data = {
            "content": question,
            "author": message.author.name,
            "timestamp": message.created_at.isoformat()
        }
        await save_message_to_vector_db(data)
        print(f"Message saved to the vector db: {data}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to save message: {e}")
        raise

async def periodic_messages(channel):
    global last_message_time
    global thread_id

    while True:
        if time.time() - last_message_time > 1 * 3600:
            last_message_time = time.time()

            try:
                random_message = await generate_intro()
                if len(random_message) > 0:
                    thread_id = uuid4()
                    await channel.send(random_message)
            except requests.exceptions.RequestException as e:
                print(f"Failed to generate intro message: {e}")
                raise  
        
        await asyncio.sleep(random.uniform(2 * 3600, 24 * 3600))

client.run(TOKEN)