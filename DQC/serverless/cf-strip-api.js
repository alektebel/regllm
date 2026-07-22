// CloudFront Function (viewer-request) attached to the `/api/*` behavior.
//
// The Angular app calls relative paths like `/api/dqc/generate`. In local
// dev, proxy.conf.json rewrites `^/api` → '' before hitting the FastAPI
// backend (whose routes live under `/dqc`, `/health`, …). This function
// reproduces that rewrite at the edge so the hosted build is byte-for-byte
// the same as local — no CORS, no frontend changes. `/api/dqc/x` → `/dqc/x`.
function handler(event) {
    var req = event.request;
    req.uri = req.uri.replace(/^\/api/, '');
    if (req.uri === '') { req.uri = '/'; }
    return req;
}
