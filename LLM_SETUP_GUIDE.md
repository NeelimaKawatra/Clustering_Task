# LLM API Setup Guide

Your Clustery application has built-in support for real LLM APIs. Here's how to set them up:

## 1. Install Dependencies

The required packages are already added to `environment.yml`. Install them with:

```bash
conda env update -f environment.yml
```

Or install manually:
```bash
pip install openai>=1.0.0 anthropic>=0.25.0
```

## 2. Set Up API Keys

### Option A: Environment Variables (Recommended)

Set your API keys as environment variables:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your_openai_api_key_here
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

### Option B: .env File

Create a `.env` file in your project root:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 3. Get API Keys

### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)

### Anthropic API Key
1. Go to https://console.anthropic.com/
2. Sign in or create an account
3. Go to API Keys section
4. Create a new key
5. Copy the key (starts with `sk-ant-`)

## 4. Using Real LLM APIs in the App

### In the Fine-tuning Tab:

1. **Navigate to the Fine-tuning tab** after completing clustering
2. **Scroll down to the "AI Assist" section**
3. **Configure the LLM settings:**
   - Select your preferred operation (Suggest Cluster Names, Suggest Entry Moves, etc.)
   - Adjust creativity/temperature (0.0-1.0)
   - Enter a model hint (optional, e.g., "gpt-4o-mini" or "claude-3-sonnet")

### Programmatically:

```python
from frontend.frontend_finetuning import get_llm_wrapper, initLLM, callLLM

# Initialize with OpenAI
success = initLLM("openai", {"model": "gpt-4o-mini"})

# Or with Anthropic
success = initLLM("anthropic", {"model": "claude-3-sonnet-20240229"})

# Make API calls
response = callLLM("Analyze these clusters and suggest better names", 
                   context={"clusters": cluster_data})
```

## 5. Supported Models

### OpenAI Models:
- `gpt-4o` (latest GPT-4)
- `gpt-4o-mini` (cheaper, faster)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Anthropic Models:
- `claude-3-5-sonnet-20241022` (latest)
- `claude-3-5-haiku-20241022` (faster, cheaper)
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

## 6. AI Operations Available

1. **Suggest Cluster Names**: AI analyzes cluster content and suggests better names
2. **Suggest Entry Moves**: AI identifies entries that should be moved to different clusters
3. **Suggest Merges/Splits**: AI recommends which clusters to merge or split
4. **Free-form Query**: Ask AI for general clustering help

## 7. Cost Considerations

- **OpenAI**: Pay per token (input + output)
  - GPT-4o-mini: ~$0.15/1M input tokens, $0.60/1M output tokens
  - GPT-4o: ~$2.50/1M input tokens, $10/1M output tokens

- **Anthropic**: Pay per token (input + output)
  - Claude-3-5-Haiku: ~$0.25/1M input tokens, $1.25/1M output tokens
  - Claude-3-5-Sonnet: ~$3/1M input tokens, $15/1M output tokens

## 8. Troubleshooting

### Common Issues:

1. **"API key not found"**: Make sure environment variables are set correctly
2. **"Package not installed"**: Run `pip install openai anthropic`
3. **"Rate limit exceeded"**: Wait a moment and try again, or check your API usage
4. **"Model not found"**: Check the model name is correct and available

### Testing Your Setup:

```python
from frontend.frontend_finetuning import quick_llm_test

# Test if LLM is working
if quick_llm_test():
    print("✅ LLM is working!")
else:
    print("❌ LLM setup issue")
```

## 9. Security Notes

- Never commit API keys to version control
- Use environment variables or secure key management
- Monitor your API usage to avoid unexpected charges
- Consider using API key restrictions in your provider's dashboard

## 10. Advanced Configuration

You can customize the LLM behavior by modifying the `LLMWrapper` class in `frontend/frontend_finetuning.py`:

- Adjust system prompts
- Change temperature ranges
- Add custom model parameters
- Implement additional providers

The current implementation is designed to be simple and direct, following Linus Torvalds' philosophy of avoiding over-engineering while providing powerful functionality.
