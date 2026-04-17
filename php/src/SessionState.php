<?php

declare(strict_types=1);

namespace HopTop\Fit;

enum SessionState: string
{
    case Init = 'init';
    case Advise = 'advise';
    case Frontier = 'frontier';
    case Score = 'score';
    case Trace = 'trace';
    case Done = 'done';
}
