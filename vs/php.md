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

// Create session with your implementations
$session = new Session(
    advisor: new RemoteAdvisor('http://localhost:8080'),
    adapter: new AnthropicAdapter($_ENV['ANTHROPIC_API_KEY']),
    scorer: new CompositeScorer(['accuracy', 'relevance', 'safety']),
);

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

Combine multiple scorers:

```php
use Hop\Fit\CompositeScorer;

$scorer = new CompositeScorer(
    scorers: [new TaxAccuracyScorer(), new SafetyScorer()],
    weights: [0.7, 0.3],
);
```

## Trace handling

```php
use Hop\Fit\TraceWriter;

$writer = new TraceWriter('./traces');
$writer->write($trace, step: 1);

// Traces stored as:
// ./traces/{session_id}/step-001.yaml
```

Traces are xrr-compatible YAML cassettes. Load for replay:

```php
$reader = new TraceReader('./traces');
$sessions = $reader->listSessions();
$data = $reader->read('sess_abc123', step: 1);
```

## Multi-turn sessions

```php
$session = new Session(
    advisor: $advisor,
    adapter: $adapter,
    scorer: $scorer,
    config: [
        'mode' => 'multi-turn',
        'max_steps' => 10,
        'reward_threshold' => 0.95,
    ],
);
```

## Laravel integration

```php
// app/Services/FitService.php
namespace App\Services;

use Hop\Fit\Session;

class FitService
{
    private Session $session;

    public function __construct()
    {
        $this->session = new Session(
            advisor: new RemoteAdvisor(config('fit.advisor_endpoint')),
            adapter: new AnthropicAdapter(config('fit.anthropic_key')),
            scorer: new CompositeScorer(['accuracy', 'safety']),
        );
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
