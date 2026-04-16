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
 * PR#17 regression: failure rewards must include metadata.error reason.
 */
class SessionMetadataRegressionTest extends TestCase
{
    public function testAdapterFailureProducesFrontierFailureMetadata(): void
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
        [, $reward] = $session->run('test prompt');

        $this->assertNull($reward->score);
        $this->assertArrayHasKey('error', $reward->metadata);
        $this->assertSame('frontier_failure', $reward->metadata['error']);
    }

    public function testScorerFailureProducesScorerFailureMetadata(): void
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
                return ['output', ['model' => 'test']];
            }
        };

        $scorer = new class implements RewardScorerInterface {
            public function score(string $output, array $context): Reward
            {
                throw new \RuntimeException('scorer exploded');
            }
        };

        $session = new Session($advisor, $adapter, $scorer);
        [, $reward] = $session->run('test prompt');

        $this->assertNull($reward->score);
        $this->assertArrayHasKey('error', $reward->metadata);
        $this->assertSame('scorer_failure', $reward->metadata['error']);
    }

    public function testSuccessfulRunHasNoErrorInRewardMetadata(): void
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
                return ['output', ['model' => 'test']];
            }
        };

        $scorer = new class implements RewardScorerInterface {
            public function score(string $output, array $context): Reward
            {
                return new Reward(0.9, ['accuracy' => 0.9]);
            }
        };

        $session = new Session($advisor, $adapter, $scorer);
        [, $reward] = $session->run('test prompt');

        $this->assertSame(0.9, $reward->score);
        $this->assertArrayNotHasKey('error', $reward->metadata);
    }
}
