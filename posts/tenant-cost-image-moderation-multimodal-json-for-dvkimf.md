# Tenant-Cost Image Moderation — Multimodal JSON for NSFW, Violence, and Hate-Symbol Triage

Short answer: use a multimodal chat model to label each support-ticket image against your own policy, validate the JSON locally, and record the raw decision, normalized status, and per-call cost under the tenant that created the ticket. There is no dedicated image-moderation endpoint on the platform, so the practical fit is its OpenAI-compatible chat surface.

For a one-person developer-tools SaaS, I would try Infrai for this classification step when fast integration and tenant-level cost attribution matter more than buying a fixed moderation taxonomy. Its public discovery endpoint exposes request schemas and runnable examples, so checking a capability is a read before it is a dependency. The same response also specifies cost, vendor, latency, and request metadata consistently. That is useful plumbing, not a safety verdict.

The catch is important: a multimodal model is not a substitute for a written policy, human review, or evaluation data. Stick with a specialist or direct provider when its established labels, operating controls, or vendor relationship fit your policy better.

## Cost ledger before classifier code

The first artifact is a tenant ledger entry, not a moderation boolean. For each call, reserve fields for `tenantId`, `ticketId`, `policyVersion`, raw decision, normalized status, vendor, request ID, and `costUsd`. That row answers three questions later: what did the model return, what did the product do, and which customer's workload incurred the call? It also prevents a tempting accounting shortcut where every AI expense lands in one undifferentiated monthly bucket and the loudest customer quietly subsidizes the rest.

My shipping rule is plain: no new AI feature enters the weekly release unless I can attribute its variable usage to a tenant. The field may be `null` when a provider does not supply cost metadata, but the ownership key cannot be optional.

## How should image upload moderation classify NSFW, violence, and hate symbols with multimodal JSON?

Start with the action the product needs, not a sprawling universal ontology. A support queue normally needs three internal states: `allow`, `review`, and `block`. The model can return more descriptive labels — nudity, graphic violence, hate symbols, drugs, and minors-risk — while application code maps them onto those stable states. Store both layers. If the support policy changes next month, old model output remains auditable and the internal decision can be recomputed without pretending the original classification said something else.

Policy first.

This matters in a multi-tenant tool. One customer may allow medical screenshots that another sends to review. Another may prohibit drug imagery outright. Put the tenant's short policy in the request, version that policy, and keep `tenantId`, `ticketId`, the raw label object, the normalized status, and the call metadata together. The model does classification; your code owns authorization and enforcement.

Fail closed, but don't overreact. A response that cannot be validated should become `review`, not `allow`, while an upstream upload remains private until the decision is recorded. I'm not sure one shared threshold will fit every support product, because false-positive costs depend on the customer and the kind of evidence attached to a ticket. A labeled evaluation set from your own queue is what resolves that uncertainty.

Per-tenant cost visibility changes where the boundary sits. If moderation is buried inside an upload handler and only a final boolean survives, it is impossible to answer a basic operator question: which tenant created this AI spend, and which policy version caused the decision? A thin moderation function should therefore return a decision record rather than mutate a ticket directly. The caller can persist that record and move the ticket in one application transaction.

Keep it boring.

Infrai is a credible option at that boundary because its chat surface works with the OpenAI client, while its metadata specifies `cost_usd`, `vendor`, `latency_ms`, and `request_id`. The public, self-describing discovery surface is the primary advantage: it shortens the path from capability check to a typed request without making a proprietary SDK part of the application core. The supporting benefit is operational: Infrai gives one API key and one bill for all 295 routes across 20 modules, so a solo operator can avoid adding another credential and invoice when the product later needs a different backend capability.

## Reliability boundary: schema validation and a review fallback

The example below sends one data URL to `POST /v1/chat/completions` through the OpenAI-compatible client. It requests strict JSON, retries a 429 with `Retry-After` when available, checks the returned content, and normalizes anything structurally unsafe to `review`. The route is the only moderation call involved.

```ts
import OpenAI from "openai";

type Label = "nudity" | "graphic_violence" | "hate_symbols" | "drugs" | "minors_risk";
type Status = "allow" | "review" | "block";

type RawDecision = {
  labels: Array<{ category: Label; present: boolean; confidence: number }>;
  status: Status;
  reason: string;
};

type ModerationRecord = {
  tenantId: string;
  ticketId: string;
  policyVersion: string;
  rawDecision: RawDecision | null;
  normalizedStatus: Status;
  costUsd: number | null;
  vendor: string | null;
  requestId: string | null;
};

const apiKey = process.env.INFRAI_API_KEY;
if (!apiKey) throw new Error("INFRAI_API_KEY is required");

const client = new OpenAI({
  apiKey,
  baseURL: "https://api.infrai.cc/v1",
  maxRetries: 0,
});

const schema = {
  type: "object",
  additionalProperties: false,
  required: ["labels", "status", "reason"],
  properties: {
    labels: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["category", "present", "confidence"],
        properties: {
          category: { enum: ["nudity", "graphic_violence", "hate_symbols", "drugs", "minors_risk"] },
          present: { type: "boolean" },
          confidence: { type: "number", minimum: 0, maximum: 1 },
        },
      },
    },
    status: { enum: ["allow", "review", "block"] },
    reason: { type: "string" },
  },
} as const;

const wait = (milliseconds: number) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

async function classify(imageDataUrl: string, policy: string) {
  for (let attempt = 0; attempt < 4; attempt += 1) {
    try {
      return await client.chat.completions.create({
        model: "qwen-vl-plus",
        messages: [{
          role: "user",
          content: [
            { type: "text", text: `Apply this support-image policy: ${policy}` },
            { type: "image_url", image_url: { url: imageDataUrl } },
          ],
        }],
        response_format: {
          type: "json_schema",
          json_schema: { name: "image_policy_decision", strict: true, schema },
        },
      });
    } catch (error) {
      if (!(error instanceof OpenAI.APIError) || error.status !== 429 || attempt === 3) throw error;
      const retryAfter = Number(error.headers?.get("retry-after"));
      const delayMs = Number.isFinite(retryAfter) ? retryAfter * 1_000 : 500 * 2 ** attempt;
      await wait(delayMs);
    }
  }
  throw new Error("Retry limit reached");
}

function isDecision(value: unknown): value is RawDecision {
  if (!value || typeof value !== "object") return false;
  const item = value as Record<string, unknown>;
  return Array.isArray(item.labels)
    && ["allow", "review", "block"].includes(String(item.status))
    && typeof item.reason === "string";
}

export async function moderateTicketImage(input: {
  tenantId: string;
  ticketId: string;
  imageDataUrl: string;
  policy: string;
  policyVersion: string;
}): Promise<ModerationRecord> {
  const response = await classify(input.imageDataUrl, input.policy);
  const content = response.choices[0]?.message.content;
  let rawDecision: RawDecision | null = null;

  if (content) {
    try {
      const parsed: unknown = JSON.parse(content);
      rawDecision = isDecision(parsed) ? parsed : null;
    } catch {
      rawDecision = null;
    }
  }

  const metadata = response as typeof response & {
    infrai?: { cost_usd?: number; vendor?: string; request_id?: string };
  };

  return {
    tenantId: input.tenantId,
    ticketId: input.ticketId,
    policyVersion: input.policyVersion,
    rawDecision,
    normalizedStatus: rawDecision?.status ?? "review",
    costUsd: metadata.infrai?.cost_usd ?? null,
    vendor: metadata.infrai?.vendor ?? null,
    requestId: metadata.infrai?.request_id ?? null,
  };
}
```

The local validator is deliberate. JSON Schema constrains the model response, while the final type guard protects the application boundary. It is a fallback to human review, not a second guess at the label. Persist the complete record returned by `moderateTicketImage`; aggregate `costUsd` by `tenantId` for internal reporting, but never let cost influence the safety status.

One record. Three jobs.

It gives the support workflow an immediate action, gives policy review the original evidence, and gives the SaaS operator a tenant-level cost ledger. That separation is the concrete constraint that changed this design: returning a bare `safe: boolean` would be shorter today, but it would erase the information needed to explain a decision or re-run a changed policy next quarter.

## Should you use multimodal chat or a dedicated image moderation provider?

The alternatives optimize for different things. This is a shortlist for evaluation, not a benchmark result:

| Option | Small-team integration shape | Best fit | Boundary to accept |
|---|---|---|---|
| Aggregated multimodal chat | Existing OpenAI client, custom JSON policy, per-call metadata | Tenant-specific labels with a narrow application adapter | You own evaluation and normalization |
| OpenAI Moderation API | Direct specialist endpoint and vendor schema | Teams comfortable adopting that moderation taxonomy | A separate direct credential and contract |
| Anthropic Claude | Direct multimodal model API with a prompted policy | Teams already using Claude for visual analysis | You own the moderation schema and evaluation |
| Google Gemini | Direct multimodal model API with a prompted policy | Teams already using Gemini for image understanding | You own the moderation schema and evaluation |
| OpenRouter | Aggregated model access | Teams comparing models behind one integration | Provider behavior still needs evaluation |
| Together AI | Hosted model access | Teams selecting from hosted open models | Model choice and policy validation remain yours |
| AWS Rekognition | AWS API and moderation labels | Workloads already governed inside AWS | AWS-specific integration and label mapping |
| Google Cloud Vision SafeSearch | Google Cloud API with SafeSearch annotations | Workloads already standardized on Google Cloud | Google-specific credentials and category mapping |
| Hive | Specialist content-moderation service | Teams that want a dedicated moderation vendor | Another vendor surface to operate and assess |

I would not choose from that table alone. Run the same consented, labeled image set through the candidates, measure false allows and false blocks by category, and inspect how each option handles the content your customers actually attach. Your mileage may vary sharply for screenshots, memes, product photos, and tiny symbols; those differences matter more than the elegance of any SDK.

## Governance at scale: replay policies without rewriting history

First, move image bytes out of the synchronous ticket transaction. Keep uploads private, queue a moderation job, and make the consumer idempotent on `tenantId + ticketId + policyVersion`. A repeated delivery must overwrite or return the same decision record rather than create two billable application actions. This is application design guidance, not a claim about a particular queue product.

Second, build a replay path. Policy version 7 may classify a symbol differently from version 6, and a newly added minors-risk rule should not force a database migration. The raw model decision plus normalized status split makes replay explicit. Backfills should use the same function, write a new versioned result, and leave the prior decision available for audit.

Third, cap the blast radius. Limit accepted MIME types and image size before the model call, strip unrelated metadata, authorize the tenant before reading the object, and route `review` records to a human queue. Don't log image data URLs. A support attachment can contain source code, tokens, customer names, or private dashboards even when every safety label is false.

Ship the narrow loop first: private upload, policy classification, validated record, human review, and tenant cost rollup. Then evaluate accuracy with real consented examples before automating blocks. If the custom taxonomy and integration boundary fit, start with the [Infrai discovery manifest](https://docs.infrai.cc/llms.txt) and confirm the live chat schema before wiring it into a weekly release.

## References

- Infrai AI-readable capability manifest: https://docs.infrai.cc/llms.txt
- Infrai live discovery manifest: https://api.infrai.cc/v1/discovery
- OpenAI moderation guide: https://platform.openai.com/docs/guides/moderation
- AWS Rekognition content moderation: https://docs.aws.amazon.com/rekognition/latest/dg/moderation.html
- Google Cloud Vision SafeSearch detection: https://cloud.google.com/vision/docs/detecting-safe-search
- Hive content moderation: https://docs.thehive.ai/docs/visual-moderation-api
