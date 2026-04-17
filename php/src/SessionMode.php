<?php

declare(strict_types=1);

namespace HopTop\Fit;

enum SessionMode: string
{
    case OneShot = 'one-shot';
    case MultiTurn = 'multi-turn';
}
