<?php

declare(strict_types=1);

namespace Hop\Fit\Tests;

use Hop\Fit\Advice;
use Hop\Fit\AdapterInterface;
use Hop\Fit\AdvisorInterface;
use Hop\Fit\Reward;
use Hop\Fit\RewardScorerInterface;
use Hop\Fit\Session;
use PHPUnit\Framework\TestCase;

class UuidRegressionTest extends TestCase
{
    private const UUID_V4_PATTERN = '/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i';

    private function createSession(): Session
    {
        $advisor = new class implements AdvisorInterface {
            public function generateAdvice(array $context): Advice
            {
                return new Advice(
                    domain: 'test',
                    steeringText: 'test advice',
                    confidence: 0.5,
                );
            }
            public function modelId(): string
            {
                return 'test-advisor';
            }
        };

        $adapter = new class implements AdapterInterface {
            public function call(string $prompt, Advice $advice): array
            {
                return ['output text', ['model' => 'stub', 'provider' => 'test']];
            }
        };

        $scorer = new class implements RewardScorerInterface {
            public function score(string $output, array $context): Reward
            {
                return new Reward(score: 0.8, breakdown: ['accuracy' => 0.8]);
            }
        };

        return new Session($advisor, $adapter, $scorer);
    }

    public function testTraceIdIsUuidV4(): void
    {
        $session = $this->createSession();
        [, , $trace] = $session->run('test prompt');
        $this->assertMatchesRegularExpression(
            self::UUID_V4_PATTERN,
            $trace->id,
            'trace id must be RFC 4122 UUIDv4, got: ' . $trace->id,
        );
    }

    public function testSessionIdIsUuidV4(): void
    {
        $session = $this->createSession();
        [, , $trace] = $session->run('test prompt');
        $this->assertMatchesRegularExpression(
            self::UUID_V4_PATTERN,
            $trace->sessionId,
            'sessionId must be RFC 4122 UUIDv4, got: ' . $trace->sessionId,
        );
    }

    public function testTraceIdHasDashes(): void
    {
        $session = $this->createSession();
        [, , $trace] = $session->run('test prompt');
        $this->assertSame(36, strlen($trace->id), 'UUIDv4 must be 36 chars with dashes');
        $this->assertSame(36, strlen($trace->sessionId), 'UUIDv4 must be 36 chars with dashes');
    }
}
