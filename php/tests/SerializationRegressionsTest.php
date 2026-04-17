<?php

declare(strict_types=1);

namespace HopTop\Fit\Tests;

use HopTop\Fit\Advice;
use HopTop\Fit\Reward;
use HopTop\Fit\Trace;
use PHPUnit\Framework\TestCase;

class SerializationRegressionsTest extends TestCase
{
    // Issue 2 regression: Trace::toArray() must use snake_case keys
    public function testTraceToArrayUsesSnakeCaseKeys(): void
    {
        $a = new Advice(domain: 'x', steeringText: 'y', confidence: 0.5);
        $r = new Reward(score: 0.9, breakdown: ['accuracy' => 1.0]);
        $t = new Trace(
            id: 'trace-1',
            sessionId: 'sess-1',
            timestamp: '2026-01-01T00:00:00Z',
            input: ['prompt' => 'test'],
            advice: $a,
            frontier: ['model' => 'stub'],
            reward: $r,
        );

        $arr = $t->toArray();

        // Must be session_id, not sessionId
        $this->assertArrayHasKey('session_id', $arr);
        $this->assertSame('sess-1', $arr['session_id']);
        $this->assertArrayNotHasKey('sessionId', $arr);
    }

    // Issue 3 regression: Advice::toArray() must use snake_case keys
    public function testAdviceToArrayUsesSnakeCaseSteeringText(): void
    {
        $a = new Advice(domain: 'tax', steeringText: 'cite sources', confidence: 0.8);

        $arr = $a->toArray();

        // Must be steering_text, not steeringText
        $this->assertArrayHasKey('steering_text', $arr);
        $this->assertSame('cite sources', $arr['steering_text']);
        $this->assertArrayNotHasKey('steeringText', $arr);
    }

    // Issue 5 regression: Session.run() must nest context under input.context
    public function testSessionInputNestsContextProperly(): void
    {
        $advisor = new class implements \HopTop\Fit\AdvisorInterface {
            public array $receivedInput = [];
            public function generateAdvice(array $input): Advice
            {
                $this->receivedInput = $input;
                return new Advice(domain: 'test', steeringText: '', confidence: 0.5);
            }
            public function modelId(): string
            {
                return 'test-advisor';
            }
        };
        $adapter = new class implements \HopTop\Fit\AdapterInterface {
            public function call(string $prompt, Advice $advice): array
            {
                return ['output', ['model' => 'test']];
            }
        };
        $scorer = new class implements \HopTop\Fit\RewardScorerInterface {
            public function score(string $output, array $context): Reward
            {
                return new Reward(score: 0.5, breakdown: []);
            }
        };

        $session = new \HopTop\Fit\Session($advisor, $adapter, $scorer);
        $context = ['jurisdiction' => 'US', 'filing_status' => 'single'];
        $session->run('What is the deduction?', $context);

        // Input to advisor must nest context, not flatten it
        $input = $advisor->receivedInput;
        $this->assertArrayHasKey('prompt', $input);
        $this->assertArrayHasKey('context', $input);
        $this->assertArrayNotHasKey('jurisdiction', $input);
        $this->assertSame('What is the deduction?', $input['prompt']);
        $this->assertSame($context, $input['context']);
    }
}
