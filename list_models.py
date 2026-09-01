"""
List the Groq models your API key can actually use.

    python3 list_models.py

Providers retire model names periodically. If generation starts failing,
run this and set the one you want:

    export GROQ_MODEL=<name from the list>
"""

import os
import sys

key = os.environ.get("GROQ_API_KEY")
if not key:
    sys.exit("No GROQ_API_KEY set. Get one free (no card) at "
             "https://console.groq.com/keys")

try:
    from groq import Groq
except ImportError:
    sys.exit("pip install groq")

client = Groq(api_key=key)

print("Models available to this key:\n")
names = []
try:
    for m in client.models.list().data:
        names.append(m.id)
        print(" ", m.id)
except Exception as e:
    sys.exit(f"Could not list models: {e}")

# Speech, safety-classifier and text-to-speech models can't do this job.
SKIP = ("whisper", "prompt-guard", "orpheus", "tts", "guard")
chat = [n for n in names if not any(k in n.lower() for k in SKIP)]
print("\nUsable for extraction / generation:\n")
for n in chat:
    print(" ", n)

import llm
print(f"\nConfigured — fast (extraction):   {llm.MODEL_FAST}")
print(f"             strong (drafting):  {llm.MODEL_STRONG}")
for label, m in (("fast", llm.MODEL_FAST), ("strong", llm.MODEL_STRONG)):
    if m in names:
        print(f"  {label}: available")
    else:
        print(f"  {label}: '{m}' NOT available. Pick one above and run:")
        env = "GROQ_MODEL_FAST" if label == "fast" else "GROQ_MODEL_STRONG"
        print(f"    export {env}=<name>")
