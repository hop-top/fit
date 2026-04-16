<?php

declare(strict_types=1);

namespace Hop\Fit;

class Session
{
    public function __construct(
        private AdvisorInterface $advisor,
        private AdapterInterface $adapter,
        private RewardScorerInterface $scorer,
    ) {}

    /**
     * @return array{string, Reward, Trace}
     */
    public function run(string $prompt, array $context = []): array
    {
        $sessionId = bin2hex(random_bytes(16));

        try {
            $advice = $this->advisor->generateAdvice(['prompt' => $prompt] + $context);
        } catch (\Throwable) {
            $advice = new Advice('unknown', '', 0.0);
        }

        [$output, $frontierMeta] = $this->adapter->call($prompt, $advice);

        try {
            $reward = $this->scorer->score($output, $context);
        } catch (\Throwable) {
            $reward = new Reward(NAN, []);
        }

        $trace = new Trace(
            id: bin2hex(random_bytes(16)),
            sessionId: $sessionId,
            timestamp: gmdate('c'),
            input: ['prompt' => $prompt] + $context,
            advice: $advice,
            frontier: $frontierMeta,
            reward: $reward,
        );

        return [$output, $reward, $trace];
    }
}
