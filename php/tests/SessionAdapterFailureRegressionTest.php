<?php

declare(strict_types=1);

namespace Hop\Fit\Tests;

use Hop\Fit\AdapterInterface;
use Hop\Fit\Advice;
use Hop\Fit\AdvisorInterface;
use Hop\Fit\Reward;
use Hop\Fit\RewardScorerInterface;
use Hop\Fit\Session;
use PHPUnit\Framework\TestCase;

/**
 * PR#12 regression: adapter->call() must be wrapped in try/catch.
 *
 * When the adapter throws, the session must return a partial trace
 * with NAN reward — not propagate the exception.
 *
 * Other ports (Go, Python, TypeScript) were fixed in PR#11.
 * PHP was missed.
 */
class SessionAdapterFailureRegressionTest extends TestCase
{
    public function testAdapterThrowingReturnsPartialTraceWithNanReward(): void
    {
        $advisor = new class implements AdvisorInterface {
            public function generateAdvice(array $context): Advice
            {
                return new Advice('test', 'steer', 0.5);
            }
            public function modelId(): string { return 'stub'; }
        };

        $adapter = new class implements AdapterInterface {
            public function call(string $prompt, Advice $advice): array
            {
                throw new \RuntimeException('adapter exploded');
            }
        };

        $scorer = new class implements RewardScorerInterface {
            public function score(string $output, array $context): Reward
            {
                return new Reward(1.0, ['accuracy' => 1.0]);
            }
        };

        $session = new Session($advisor, $adapter, $scorer);

        $result = $session->run('test prompt');

        // Must return a 3-tuple, not throw
        $this->assertIsArray($result);
        $this->assertCount(3, $result);

        [$output, $reward, $trace] = $result;

        // Output should be empty string on adapter failure
        $this->assertSame('', $output);

        // Reward score must be NAN (not a real score)
        $this->assertNan($reward->score);

        // Trace must exist with partial data
        $this->assertNotNull($trace);
        $this->assertSame('test prompt', $trace->input['prompt']);
        $this->assertArrayHasKey('error', $trace->frontier);
        $this->assertSame('adapter exploded', $trace->frontier['error']);
    }
}
