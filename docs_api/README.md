
# API Documentation (Swagger UI)

This folder hosts the OpenAPI specification for the Data Transfer API. The file
`specification-api.json` defines the endpoints, request/response schemas, and
authentication requirements used by the service. The `index.html` file loads the
spec into Swagger UI so you can explore and test the API in your browser.

## Run locally

Serve this folder with a simple HTTP server and then open the Swagger UI:

```bash
python3 -m http.server 8080
```

Then visit:

- http://localhost:8080/index.html

