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
 * PR#12 regression: adapter->call() must be wrapped in try/catch.
 *
 * When the adapter throws, the session must return a partial trace
 * with null reward score — not propagate the exception.
 *
 * PR#15: changed from NAN to null per reward-schema-v1.
 */
class SessionAdapterFailureRegressionTest extends TestCase
{
    public function testAdapterThrowingReturnsPartialTraceWithNullScore(): void
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

        // Reward score must be null per reward-schema-v1 (not NAN)
        $this->assertNull($reward->score);

        // Trace must exist with partial data
        $this->assertNotNull($trace);
        $this->assertSame('test prompt', $trace->input['prompt']);
        $this->assertArrayHasKey('error', $trace->frontier);
        $this->assertSame('adapter exploded', $trace->frontier['error']);
    }

    public function testValidEmptyOutputNotMisclassifiedAsFailure(): void
    {
        // PR#15 regression: valid empty-string output must not be
        // misclassified as adapter failure.
        $advisor = new class implements AdvisorInterface {
            public function generateAdvice(array $context): Advice
            {
                return new Advice('test', 'steer', 0.5);
            }
            public function modelId(): string { return 'stub'; }
        };

        // Adapter returns empty string as valid output (no error in frontier)
        $adapter = new class implements AdapterInterface {
            public function call(string $prompt, Advice $advice): array
            {
                return ['', ['model' => 'test']];
            }
        };

        $scorer = new class implements RewardScorerInterface {
            public function score(string $output, array $context): Reward
            {
                return new Reward(0.5, ['accuracy' => 0.5]);
            }
        };

        $session = new Session($advisor, $adapter, $scorer);
        [, $reward, $trace] = $session->run('test');

        // Must use scorer result, not null failure reward
        $this->assertSame(0.5, $reward->score);
        $this->assertArrayNotHasKey('error', $trace->frontier);
    }

    public function testNullScoreRoundTripsThroughJson(): void
    {
        // PR#15 regression: Reward with null score must serialize
        // to valid JSON (score: null, not NaN).
        $reward = new Reward(null, []);
        $arr = $reward->toArray();

        $json = json_encode($arr);
        $this->assertNotFalse($json, 'json_encode must succeed');

        $decoded = json_decode($json, true);
        $this->assertNull($decoded['score']);

        // NaN would cause json_encode to fail or produce invalid JSON
        $nanReward = ['score' => NAN];
        $nanJson = json_encode($nanReward);
        $this->assertFalse($nanJson, 'NAN must fail json_encode');
    }
}
