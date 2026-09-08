# Helpers for driving the live Spinwave synth over its TCP control channel.
# Dot-source, then: Send-Live '{"cmd":"ping"}' ; Push-Patch path.vital ; Play-Seq @(...)

function Send-Live([string]$json, [int]$port = 41929) {
    $client = New-Object System.Net.Sockets.TcpClient('127.0.0.1', $port)
    try {
        $stream = $client.GetStream()
        $writer = New-Object System.IO.StreamWriter($stream)
        $writer.NewLine = "`n"
        $writer.WriteLine($json)
        $writer.Flush()
        $reader = New-Object System.IO.StreamReader($stream)
        $reader.ReadLine()
    } finally {
        $client.Close()
    }
}

function Push-Patch([string]$presetPath) {
    $preset = Get-Content $presetPath -Raw | ConvertFrom-Json
    $message = @{ cmd = 'preset'; preset = $preset } | ConvertTo-Json -Depth 24 -Compress
    Send-Live $message
}

# Events: array of @{t=<ms>; on=<note>} / @{t=<ms>; off=<note>} / @{t=<ms>; v=<velocity>}
function Play-Seq([array]$events) {
    $velocity = 0.85
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    foreach ($e in ($events | Sort-Object { $_.t })) {
        $wait = $e.t - $clock.ElapsedMilliseconds
        if ($wait -gt 0) { Start-Sleep -Milliseconds $wait }
        if ($null -ne $e.v) { $velocity = $e.v }
        elseif ($null -ne $e.on) {
            Send-Live ('{"cmd":"note_on","note":' + $e.on + ',"velocity":' + $velocity + '}') | Out-Null
        }
        elseif ($null -ne $e.off) {
            Send-Live ('{"cmd":"note_off","note":' + $e.off + '}') | Out-Null
        }
    }
}

function Set-PatchParams([string]$presetPath, [hashtable]$params) {
    $preset = Get-Content $presetPath -Raw | ConvertFrom-Json
    foreach ($key in $params.Keys) {
        if ($preset.settings.PSObject.Properties[$key]) {
            $preset.settings.$key = $params[$key]
        } else {
            $preset.settings | Add-Member -NotePropertyName $key -NotePropertyValue $params[$key]
        }
    }
    $preset | ConvertTo-Json -Depth 24 | Set-Content -Encoding utf8 $presetPath
    Push-Patch $presetPath
}
