# Deploying Pipecat to Modal.com

Barebones deployment example for [modal.com](https://www.modal.com)

1. Setup a Modal account and install it on your machine if you have not already, following their easy 3-steps in their [Getting Started Guide](https://modal.com/docs/guide#getting-started)


2. Setup .env

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

## Configuration options

This app sets some sensible defaults for reducing cold starts, such as `min_containers=1`, which will keep at least 1 warm instance ready for your bot function.

It has been configured to only allow a concurrency of 1 (`max_inputs=1`) as each user will require their own running function.