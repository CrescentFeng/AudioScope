<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

function respond(mixed $payload, int $status = 200): never {
    http_response_code($status);
    echo json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    exit;
}

function connection(): PDO {
    $host = getenv('AUDIOSCOPE_DB_HOST') ?: 'db';
    $port = getenv('AUDIOSCOPE_DB_PORT') ?: '5432';
    $name = getenv('AUDIOSCOPE_DB_NAME') ?: 'audioscope';
    $user = getenv('AUDIOSCOPE_DB_USER') ?: 'audioscope';
    $password = getenv('AUDIOSCOPE_DB_PASSWORD') ?: '';
    return new PDO(
        "pgsql:host={$host};port={$port};dbname={$name}",
        $user,
        $password,
        [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]
    );
}

try {
    $db = connection();
    $path = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH);
    if ($path === '/health') {
        $revision = $db->query('SELECT version_num FROM public.alembic_version LIMIT 1')->fetchColumn();
        respond(['status' => 'ok', 'database' => 'ready', 'revision' => $revision]);
    }
    if ($path === '/v1/readiness') {
        $tables = [
            'datasets' => 'catalog.datasets',
            'audio_files' => 'catalog.audio_files',
            'models' => 'catalog.model_artifacts',
            'benchmarks' => 'catalog.benchmark_results',
        ];
        $counts = [];
        foreach ($tables as $key => $table) {
            $counts[$key] = (int) $db->query("SELECT count(*) FROM {$table}")->fetchColumn();
        }
        $ready = $counts['datasets'] >= 2 && $counts['audio_files'] >= 3153 && $counts['models'] >= 1;
        respond(['status' => $ready ? 'ready' : 'incomplete', 'counts' => $counts]);
    }
    if ($path === '/v1/datasets') {
        $sql = 'SELECT d.slug, d.display_name, d.task_type, v.version_key, '
             . 'v.record_count, v.manifest_sha256 FROM catalog.datasets d '
             . 'JOIN catalog.dataset_versions v ON v.dataset_id = d.id ORDER BY d.slug';
        respond($db->query($sql)->fetchAll(PDO::FETCH_ASSOC));
    }
    if ($path === '/v1/models') {
        $sql = 'SELECT id, name, stage, format, storage_uri, sha256, byte_size '
             . 'FROM catalog.model_artifacts ORDER BY stage, name LIMIT 100';
        respond($db->query($sql)->fetchAll(PDO::FETCH_ASSOC));
    }
    if ($path === '/v1/benchmarks') {
        $sql = 'SELECT metric_name AS metric, metric_value AS value, unit, '
             . 'slice_name AS slice, model_artifact_id FROM catalog.benchmark_results '
             . 'ORDER BY metric_name LIMIT 200';
        respond($db->query($sql)->fetchAll(PDO::FETCH_ASSOC));
    }
    respond(['status' => 'not_found', 'path' => $path], 404);
} catch (Throwable $error) {
    respond(['status' => 'error', 'error_type' => get_class($error)], 503);
}
