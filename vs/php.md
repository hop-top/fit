# PHP: Installing via Composer

## Installation

```bash
composer require hop/fit
```

Requires PHP 8.3+.

## Basic usage

```php
use Hop\Fit\{Session, Advice, Reward, Trace};
use Hop\Fit\{AdvisorInterface, AdapterInterface, RewardScorerInterface};
use Hop\Fit\CompositeScorer;

// Implement AdvisorInterface for your advisor backend
$advisor = new MyRemoteAdvisor('http://localhost:8080');

// Implement AdapterInterface for your frontier LLM
$adapter = new MyAnthropicAdapter($_ENV['ANTHROPIC_API_KEY']);

// Combine scorers with weights (or use CompositeScorer::fromDimensions)
$scorer = new CompositeScorer(
    scorers: [new TaxAccuracyScorer(), new SafetyScorer()],
    weights: [0.7, 0.3],
);

$session = new Session($advisor, $adapter, $scorer);

[$output, $reward, $trace] = $session->run(
    'What is the standard deduction?',
    ['jurisdiction' => 'US', 'filing_status' => 'single'],
);

echo "Output: {$output}\n";
echo "Reward: {$reward->score}\n";
```

## Adapter configuration

Implement `AdapterInterface` for your frontier LLM:

```php
use Hop\Fit\{Advice, AdapterInterface};

class AnthropicAdapter implements AdapterInterface
{
    public function __construct(
        private string $apiKey,
        private string $model = 'claude-sonnet-4-6',
    ) {}

    public function call(string $prompt, Advice $advice): array
    {
        $system = "[Advisor Guidance]\n{$advice->steeringText}";

        $response = Http::post('https://api.anthropic.com/v1/messages', [
            'model' => $this->model,
            'max_tokens' => 4096,
            'system' => $system,
            'messages' => [['role' => 'user', 'content' => $prompt]],
        ]);

        $output = $response->json('content.0.text', '');
        $meta = [
            'model' => $this->model,
            'provider' => 'anthropic',
            'output' => $output,
            'usage' => $response->json('usage', []),
        ];

        return [$output, $meta];
    }
}
```

The adapter injects advice as hidden system context. End users never
see advisor guidance in the output.

## Custom reward functions

Implement `RewardScorerInterface`:

```php
use Hop\Fit\{Reward, RewardScorerInterface};

class TaxAccuracyScorer implements RewardScorerInterface
{
    public function score(string $output, array $context): Reward
    {
        $accuracy = $this->computeAccuracy($output, $context);

        return new Reward(
            score: $accuracy,
            breakdown: [
                'accuracy' => $accuracy,
                'relevance' => 0.9,
                'safety' => 1.0,
                'efficiency' => 0.8,
            ],
        );
    }

    private function computeAccuracy(string $output, array $context): float
    {
        // Domain-specific scoring logic
        return 0.85;
    }
}
```

Combine multiple scorers (takes `RewardScorerInterface[]`, not strings):

```php
use Hop\Fit\CompositeScorer;

// With explicit weights
$scorer = new CompositeScorer(
    scorers: [new TaxAccuracyScorer(), new SafetyScorer()],
    weights: [0.7, 0.3],
);

// Or via dimension names (equal weights, uses DimensionScorer internally)
$scorer = CompositeScorer::fromDimensions(['accuracy', 'relevance', 'safety']);
```

## Trace handling

```php
use Hop\Fit\TraceWriter;

$writer = new TraceWriter('./traces');
$writer->write($trace, step: 1);

// Traces stored as:
// ./traces/{session_id}/step-001.yaml
```

Traces are xrr-compatible YAML cassettes (via `symfony/yaml`).
Read trace files directly for replay or advisor training:

```php
use Symfony\Component\Yaml\Yaml;

$traceData = Yaml::parseFile('./traces/session-abc/step-001.yaml');
```

## Multi-turn sessions

```php
// Session constructor: new Session(AdvisorInterface, AdapterInterface, RewardScorerInterface)
// Configure via setter or by extending Session
$session = new Session($advisor, $adapter, $scorer);
```

## Laravel integration

```php
// app/Services/FitService.php
namespace App\Services;

use Hop\Fit\Session;
use Hop\Fit\CompositeScorer;

class FitService
{
    private Session $session;

    public function __construct()
    {
        // Use your implementations of AdvisorInterface, AdapterInterface
        $advisor = new MyRemoteAdvisor(config('fit.advisor_endpoint'));
        $adapter = new MyAnthropicAdapter(config('fit.anthropic_key'));
        $scorer = CompositeScorer::fromDimensions(['accuracy', 'safety']);

        $this->session = new Session($advisor, $adapter, $scorer);
    }

    public function ask(string $prompt, array $context = []): array
    {
        [$output, $reward, $trace] = $this->session->run($prompt, $context);
        return [
            'output' => $output,
            'score' => $reward->score,
        ];
    }
}
```

Register in a service provider:

```php
$this->app->singleton(FitService::class);
```
