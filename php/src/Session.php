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

    private static function uuidv4(): string
    {
        $data = random_bytes(16);
        $data[6] = chr((ord($data[6]) & 0x0f) | 0x40);
        $data[8] = chr((ord($data[8]) & 0x3f) | 0x80);
        return vsprintf('%s%s-%s-%s-%s-%s%s%s', str_split(bin2hex($data), 4));
    }

    /**
     * @return array{string, Reward, Trace}
     */
    public function run(string $prompt, array $context = []): array
    {
        $sessionId = self::uuidv4();

        try {
            $advice = $this->advisor->generateAdvice(['prompt' => $prompt, 'context' => $context]);
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
            id: self::uuidv4(),
            sessionId: $sessionId,
            timestamp: gmdate('c'),
            input: ['prompt' => $prompt, 'context' => $context],
            advice: $advice,
            frontier: $frontierMeta,
            reward: $reward,
        );

        return [$output, $reward, $trace];
    }
}
