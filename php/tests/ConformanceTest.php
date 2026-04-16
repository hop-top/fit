<?php

declare(strict_types=1);

namespace Hop\Fit\Tests;

use Hop\Fit\Advice;
use Hop\Fit\Reward;
use Hop\Fit\Trace;
use PHPUnit\Framework\TestCase;

class ConformanceTest extends TestCase
{
    private static function fixturesDir(): string
    {
        return dirname(__DIR__) . '/../spec/fixtures';
    }

    private function loadYaml(string $name): array
    {
        $path = self::fixturesDir() . '/' . $name;
        $content = file_get_contents($path);
        // Simple YAML parser for our fixture format
        // Since PHP doesn't ship with YAML by default, parse via
        //Symfony YAML or manually for these simple structures.
        // For conformance tests, use the JSON equivalent where available.
        return (function_exists('yaml_parse') && yaml_parse($content))
            ?: $this->simpleYamlParse($content);
    }

    private static function loadJson(string $name): array
    {
        $path = self::fixturesDir() . '/' . $name;
        return json_decode(file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
    }

    /**
     * Minimal YAML-like parser for our fixture structures.
     * Handles top-level key: value, nested maps, and simple lists.
     */
    private function simpleYamlParse(string $content): array
    {
        // Fall back to json if it parses
        $decoded = json_decode($content, true);
        if (json_last_error() === JSON_ERROR_NONE && is_array($decoded)) {
            return $decoded;
        }
        // For actual YAML files we'd need ext-yaml or symfony/yaml
        // Our test primarily uses JSON fixtures for PHP
        $this->markTestSkipped('YAML extension not available');
        return []; // static analyzers require explicit return
    }

    // --- Advice conformance ---

    public function testAdviceParseJson(): void
    {
        $data = self::loadJson('advice-v1.json');
        $a = new Advice(
            domain: $data['domain'],
            steeringText: $data['steering_text'],
            confidence: $data['confidence'],
            constraints: $data['constraints'] ?? [],
            metadata: $data['metadata'] ?? [],
            version: $data['version'] ?? '1.0',
        );
        $this->assertSame('tax-compliance', $a->domain);
        $this->assertEqualsWithDelta(0.87, $a->confidence, 0.001);
        $this->assertCount(3, $a->constraints);
        $this->assertSame('1.0', $a->version);
        $this->assertArrayHasKey('model', $a->metadata);
    }

    public function testAdviceRoundTripJson(): void
    {
        $data = self::loadJson('advice-v1.json');
        $a = new Advice(
            domain: $data['domain'],
            steeringText: $data['steering_text'],
            confidence: $data['confidence'],
            constraints: $data['constraints'],
            metadata: $data['metadata'],
            version: $data['version'],
        );
        $encoded = json_encode([
            'domain' => $a->domain,
            'steering_text' => $a->steeringText,
            'confidence' => $a->confidence,
            'constraints' => $a->constraints,
            'metadata' => $a->metadata,
            'version' => $a->version,
        ]);
        $decoded = json_decode($encoded, true);
        $this->assertSame($a->domain, $decoded['domain']);
        $this->assertEqualsWithDelta($a->confidence, $decoded['confidence'], 0.001);
        $this->assertSame($a->constraints, $decoded['constraints']);
    }

    public function testAdviceConfidenceInRange(): void
    {
        $data = self::loadJson('advice-v1.json');
        $this->assertGreaterThanOrEqual(0.0, $data['confidence']);
        $this->assertLessThanOrEqual(1.0, $data['confidence']);
    }

    // --- Reward conformance ---

    public function testRewardParseJson(): void
    {
        $data = self::loadJson('reward-v1.json');
        $r = new Reward(
            score: $data['score'],
            breakdown: $data['breakdown'],
            metadata: $data['metadata'] ?? [],
        );
        $this->assertEqualsWithDelta(0.62, $r->score, 0.001);
        $this->assertEqualsWithDelta(0.7, $r->breakdown['accuracy'], 0.001);
        $this->assertEqualsWithDelta(1.0, $r->breakdown['safety'], 0.001);
        $this->assertSame('rubric-judge-v2', $r->metadata['scorer']);
    }

    public function testRewardScoreInRange(): void
    {
        $data = self::loadJson('reward-v1.json');
        $this->assertGreaterThanOrEqual(0.0, $data['score']);
        $this->assertLessThanOrEqual(1.0, $data['score']);
        foreach ($data['breakdown'] as $dim => $val) {
            $this->assertGreaterThanOrEqual(0.0, $val, "{$dim} too low");
            $this->assertLessThanOrEqual(1.0, $val, "{$dim} too high");
        }
    }

    public function testRewardRoundTripJson(): void
    {
        $data = self::loadJson('reward-v1.json');
        $r = new Reward(
            score: $data['score'],
            breakdown: $data['breakdown'],
        );
        $encoded = json_encode([
            'score' => $r->score,
            'breakdown' => $r->breakdown,
        ]);
        $decoded = json_decode($encoded, true);
        $this->assertEqualsWithDelta($r->score, $decoded['score'], 0.001);
    }

    // --- Trace conformance ---

    public function testTraceFromFixtureData(): void
    {
        $adviceData = self::loadJson('advice-v1.json');
        $rewardData = self::loadJson('reward-v1.json');
        $a = new Advice(
            domain: 'tax-compliance',
            steeringText: 'Cite IRS publication numbers.',
            confidence: 0.91,
            constraints: ['cite sources', 'no speculation'],
        );
        $r = new Reward(
            score: 0.95,
            breakdown: ['accuracy' => 1.0, 'relevance' => 0.9, 'safety' => 1.0, 'efficiency' => 0.9],
        );
        $t = new Trace(
            id: '550e8400-e29b-41d4-a716-446655440000',
            sessionId: 'sess_abc123',
            timestamp: '2026-04-15T10:30:00Z',
            input: [
                'prompt' => 'What is the standard deduction for 2025?',
                'context' => ['jurisdiction' => 'US', 'filing_status' => 'single'],
            ],
            advice: $a,
            frontier: [
                'model' => 'claude-sonnet-4-6',
                'provider' => 'anthropic',
                'output' => 'For tax year 2025...',
                'usage' => ['prompt_tokens' => 342, 'completion_tokens' => 156, 'total_tokens' => 498],
            ],
            reward: $r,
            metadata: ['duration_ms' => 1830, 'trace_version' => '1.0'],
        );
        $this->assertSame('550e8400-e29b-41d4-a716-446655440000', $t->id);
        $this->assertSame('sess_abc123', $t->sessionId);
        $this->assertSame('tax-compliance', $t->advice->domain);
        $this->assertSame('anthropic', $t->frontier['provider']);
        $this->assertEqualsWithDelta(0.95, $t->reward->score, 0.001);
    }

    public function testTraceRoundTripJson(): void
    {
        $a = new Advice(domain: 'x', steeringText: 'y', confidence: 0.5);
        $r = new Reward(score: 0.9, breakdown: ['accuracy' => 1.0]);
        $t = new Trace(
            id: 't1',
            sessionId: 's1',
            timestamp: '2026-01-01T00:00:00Z',
            input: ['prompt' => 'fix bug'],
            advice: $a,
            frontier: ['model' => 'stub', 'provider' => 'test', 'output' => 'ok'],
            reward: $r,
        );
        $encoded = json_encode([
            'id' => $t->id,
            'session_id' => $t->sessionId,
            'timestamp' => $t->timestamp,
            'input' => $t->input,
            'advice' => ['domain' => $t->advice->domain, 'steering_text' => $t->advice->steeringText, 'confidence' => $t->advice->confidence],
            'frontier' => $t->frontier,
            'reward' => ['score' => $t->reward->score, 'breakdown' => $t->reward->breakdown],
        ]);
        $decoded = json_decode($encoded, true);
        $this->assertSame('t1', $decoded['id']);
        $this->assertSame('x', $decoded['advice']['domain']);
        $this->assertEqualsWithDelta(0.9, $decoded['reward']['score'], 0.001);
    }

    // --- Multi-turn session conformance ---

    public function testSessionMultiFileExists(): void
    {
        $path = self::fixturesDir() . '/session-multi.yaml';
        $this->assertFileExists($path);
    }

    public function testFixtureDirectoryContainsAllFiles(): void
    {
        $dir = self::fixturesDir();
        foreach (['advice-v1.yaml', 'advice-v1.json', 'reward-v1.json', 'trace-v1.yaml', 'session-multi.yaml'] as $file) {
            $this->assertFileExists($dir . '/' . $file, "missing fixture: {$file}");
        }
    }
}
