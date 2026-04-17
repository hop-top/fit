<?php

declare(strict_types=1);

namespace HopTop\Fit\Tests;

use HopTop\Fit\AdapterInterface;
use HopTop\Fit\Advice;
use HopTop\Fit\AdvisorInterface;
use HopTop\Fit\Reward;
use HopTop\Fit\RewardScorerInterface;
use HopTop\Fit\Session;
use PHPUnit\Framework\TestCase;

/**
 * PR#19 regression: spec/trace-format-v1.md requires frontier.output.
 *
 * The session must inject adapter output into frontierMeta even when
 * the adapter omits it from its metadata array.
 */
class FrontierOutputRegressionTest extends TestCase
{
    private function stubAdvisor(): AdvisorInterface
    {
        return new class implements AdvisorInterface {
            public function generateAdvice(array $context): Advice
            {
                return new Advice('test', 'steer', 0.5);
            }
            public function modelId(): string { return 'stub'; }
        };
    }

    private function stubScorer(): RewardScorerInterface
    {
        return new class implements RewardScorerInterface {
            public function score(string $output, array $context): Reward
            {
                return new Reward(0.8, ['accuracy' => 0.8]);
            }
        };
    }

    public function testFrontierOutputInjectedWhenAdapterOmitsFromMeta(): void
    {
        $adapter = new class implements AdapterInterface {
            public function call(string $prompt, Advice $advice): array
            {
                // Returns output but meta has no 'output' key.
                return ['hello', ['model' => 'test']];
            }
        };

        $session = new Session($this->stubAdvisor(), $adapter, $this->stubScorer());
        [, , $trace] = $session->run('test');

        $this->assertArrayHasKey('output', $trace->frontier);
        $this->assertSame('hello', $trace->frontier['output']);
    }

    public function testFrontierOutputInjectedOnAdapterFailure(): void
    {
        $adapter = new class implements AdapterInterface {
            public function call(string $prompt, Advice $advice): array
            {
                throw new \RuntimeException('boom');
            }
        };

        $session = new Session($this->stubAdvisor(), $adapter, $this->stubScorer());
        [, , $trace] = $session->run('test');

        // Even on failure, frontier must contain output (empty string).
        $this->assertArrayHasKey('output', $trace->frontier);
        $this->assertSame('', $trace->frontier['output']);
    }
}
