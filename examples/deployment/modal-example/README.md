# Deploying Pipecat to Modal.com

Barebones deployment example for [modal.com](https://www.modal.com)

1. Setup a Modal account and install it on your machine if you have not already, following their easy 3-steps in their [Getting Started Guide](https://modal.com/docs/guide#getting-started)


2. Navigate to the `server` directory:

```bash
cd server
```

3. Setup .env

```bash
cp env.example .env
```

Alternatively, you can configure your Modal app to use [secrets](https://modal.com/docs/guide/secrets)

3. Test the app locally

```bash
modal serve app.py
```

4. Deploy to production

```bash
modal deploy app.py
```

# Deploy a self-serve LLM

1. Follow the Modal Guide and example for [Deploying an OpenAI-compatible LLM service with vLLM](https://modal.com/docs/examples/vllm_inference).

    The TLDR; set up, though is simply to do the following in this directory:

   ```
   git clone https://github.com/modal-labs/modal-examples
   cd modal-examples
   modal deploy 06_gpu_and_ml/llm-serving/vllm_inference.py
   ```

2. Update `server/src/bot_vllm.py`
   1. Update `modal_url` to point to the url produced from the deploy in the previous step.
   2. Update `'super-secret-key'` if you have changed the API key for your llm endpoint.

# Launch and Talk to your Bots running on Modal

## Option 1: Direct Link

Simply click on the url displayed after running the server or deploy step to launch an agent and be redirected to a Daily room to talk with the launched bot.

## Option 2: Connect via an RTVI Client

Check out the [README](client/javascript/README.md) in the client folder for building and running a custom client that connects to your Modal endpoint.

# Navigate your server and Pipecat logs

In your [Modal dashboard](https://modal.com/apps), click on the  `pipecat-modal` listed under Live Apps. This will take you to the Overview of your Modal example App and will list two App Functions:
    1. fastapi_app: This function is running the endpoints that your client will interact with and initiate starting a new pipeline. Click on this function to see logs for each endpoint hit.
    2. bot_runner: This function handles launching new bot pipeline. Click on this function to get a list of all pipeline runs and access each run's logs.

# Modal + Pipecat Recommended Settings

<FIX ME: fill in the following>

<Recommended image settings for webapp container>
<Recommended image settings for pipeline container>
<Recommendations for min_containers and fast bot joins>
<Link to Advanced example with Services self-hosted on Modal>