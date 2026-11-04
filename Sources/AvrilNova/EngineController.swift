import Foundation

enum EngineError: LocalizedError {
    case missingRuntime
    case missingAlgorithm
    case launchFailed(String)
    case invalidResponse
    case server(String)

    var errorDescription: String? {
        switch self {
        case .missingRuntime: "No compatible Python science runtime was found."
        case .missingAlgorithm: "No Avril V3 engine directory was found."
        case .launchFailed(let reason): "The local engine could not start: \(reason)"
        case .invalidResponse: "The local engine returned an invalid response."
        case .server(let message): message
        }
    }
}

@MainActor
final class EngineController {
    private struct RuntimeLocations {
        let python: URL
        let algorithm: URL
        let pythonHome: URL?
    }

    private var process: Process?
    private var logHandle: FileHandle?
    private(set) var baseURL: URL?
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    func start() async throws -> String {
        if let baseURL, let health = try? await health(at: baseURL) {
            return health.version
        }
        stop()

        let locations = try locateRuntime()
        let port = Int.random(in: 53100...63900)
        let url = URL(string: "http://127.0.0.1:\(port)")!
        let dataRoot = try applicationSupportDirectory()
        let webRoot = dataRoot.appendingPathComponent("static", isDirectory: true)
        try FileManager.default.createDirectory(at: webRoot, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = locations.python
        process.arguments = [
            locations.algorithm.appendingPathComponent("desktop_bridge.py").path,
            "--port", String(port),
            "--parent-pid", String(ProcessInfo.processInfo.processIdentifier),
            "--web-root", webRoot.path,
        ]
        process.currentDirectoryURL = locations.algorithm

        var environment = ProcessInfo.processInfo.environment
        if let pythonHome = locations.pythonHome {
            environment["PYTHONHOME"] = pythonHome.path
        } else {
            environment.removeValue(forKey: "PYTHONHOME")
        }
        environment["PYTHONPATH"] = locations.algorithm.path
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        let scienceDataRoot = dataRoot.appendingPathComponent("science-data", isDirectory: true).path
        // The Avril V3 JobStore reads AVRIL_DATA_DIR.  Keep the historical
        // VLORA name as a compatibility alias for older engine entry points,
        // but never let an unset variable fall back to an unrelated external
        // archive when the workstation starts.
        environment["AVRIL_DATA_DIR"] = scienceDataRoot
        environment["VLORA_DATA_DIR"] = scienceDataRoot
        environment["MPLCONFIGDIR"] = dataRoot.appendingPathComponent("matplotlib", isDirectory: true).path
        environment["LIGHTKURVE_CACHE_DIR"] = dataRoot.appendingPathComponent("lightkurve", isDirectory: true).path
        environment["HOME"] = FileManager.default.homeDirectoryForCurrentUser.path
        process.environment = environment

        let logURL = dataRoot.appendingPathComponent("engine.log")
        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let logHandle = try FileHandle(forWritingTo: logURL)
        try logHandle.seekToEnd()
        // Keep the request-serving process independent from the GUI's stdout
        // lifecycle.  Python's threaded server may otherwise inherit a Cocoa
        // stream that stops draining while the app is launching or relaying
        // scene updates.  Diagnostics still go to the persistent error log.
        process.standardOutput = FileHandle.nullDevice
        process.standardError = logHandle

        do {
            try process.run()
        } catch {
            throw EngineError.launchFailed(error.localizedDescription)
        }
        self.process = process
        self.logHandle = logHandle
        self.baseURL = url

        for _ in 0..<160 {
            if !process.isRunning {
                throw EngineError.launchFailed("the science process exited during startup")
            }
            if let health = try? await health(at: url) {
                return health.version
            }
            try await Task.sleep(for: .milliseconds(150))
        }
        stop()
        throw EngineError.launchFailed("the startup health check timed out")
    }

    func stop() {
        if let process, process.isRunning {
            process.terminate()
        }
        try? logHandle?.close()
        process = nil
        logHandle = nil
        baseURL = nil
    }

    func currentVersionIfHealthy() async -> String? {
        guard let baseURL, let response = try? await health(at: baseURL) else { return nil }
        return response.version
    }

    func targets(offset: Int, limit: Int = 200) async throws -> TargetPage {
        try await get("/api/desktop/targets?lane=all&sort=rank&order=asc&limit=\(limit)&offset=\(offset)")
    }

    func createJob(ticID: Int, source: String, pixels: Bool) async throws -> JobDTO {
        // Avril's public job API resolves `target_number` first. A bare numeric
        // value is intentionally rejected by the science engine, so always
        // send the canonical TIC identifier in both compatibility fields.
        let canonicalTarget = "TIC \(ticID)"
        let payload: [String: String] = [
            "target_number": canonicalTarget,
            "target": canonicalTarget,
            "source": source,
            "pixels": pixels ? "followup" : "none",
            "backend": "auto",
            "ffi_engine": "core",
        ]
        return try await request("/api/jobs", method: "POST", body: encoder.encode(payload))
    }

    func job(id: String) async throws -> JobDTO {
        try await get("/api/jobs/\(id)")
    }

    func lightCurve(jobID: String) async throws -> LightCurvePayload {
        try await get("/api/desktop/jobs/\(jobID)/lightcurve")
    }

    func runRecords() async throws -> RunRecordPage {
        try await get("/api/desktop/jobs")
    }

    func executionLog(jobID: String) async throws -> ExecutionLogDocument {
        let data = try await responseData("/api/desktop/jobs/\(jobID)/events", method: "GET", body: nil)
        return try await Task.detached(priority: .userInitiated) {
            try JSONDecoder().decode(ExecutionLogDocument.self, from: data)
        }.value
    }

    func scienceTimeline(jobID: String) async throws -> ScienceTimelineDocument {
        let data = try await responseData("/api/desktop/jobs/\(jobID)/timeline", method: "GET", body: nil)
        return try await Task.detached(priority: .userInitiated) {
            try JSONDecoder().decode(ScienceTimelineDocument.self, from: data)
        }.value
    }

    func scienceResult(jobID: String) async throws -> ScienceResultDocument {
        let data = try await responseData("/api/jobs/\(jobID)/result", method: "GET", body: nil)
        return try await Task.detached(priority: .userInitiated) {
            try JSONDecoder().decode(ScienceResultDocument.self, from: data)
        }.value
    }

    func candidateDetail(jobID: String, candidateID: String) async throws -> ResultCandidate {
        guard let encoded = candidateID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else {
            throw EngineError.invalidResponse
        }
        let data = try await responseData("/api/desktop/jobs/\(jobID)/candidates/\(encoded)", method: "GET", body: nil)
        return try await Task.detached(priority: .userInitiated) {
            try JSONDecoder().decode(ResultCandidate.self, from: data)
        }.value
    }

    func candidateFolds(jobID: String, candidateID: String) async throws -> CandidateFoldDocument {
        guard let encoded = candidateID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else {
            throw EngineError.invalidResponse
        }
        let data = try await responseData(
            "/api/desktop/jobs/\(jobID)/candidates/\(encoded)/folds",
            method: "GET",
            body: nil
        )
        return try await Task.detached(priority: .userInitiated) {
            try JSONDecoder().decode(CandidateFoldDocument.self, from: data)
        }.value
    }

    func resultDownloadURL(jobID: String) -> URL? {
        guard let baseURL else { return nil }
        return URL(string: "/api/jobs/\(jobID)/result", relativeTo: baseURL)?.absoluteURL
    }

    func auditDownloadURL(jobID: String) -> URL? {
        guard let baseURL else { return nil }
        return URL(string: "/api/jobs/\(jobID)/audit", relativeTo: baseURL)?.absoluteURL
    }

    func resultArtifactData(jobID: String) async throws -> Data {
        try await responseData("/api/jobs/\(jobID)/result", method: "GET", body: nil)
    }

    func auditArtifactData(jobID: String) async throws -> Data {
        try await responseData("/api/jobs/\(jobID)/audit", method: "GET", body: nil)
    }

    func cancel(jobID: String, token: String) async throws -> JobDTO {
        let payload = try encoder.encode(["control_token": token])
        return try await request("/api/jobs/\(jobID)/cancel", method: "POST", body: payload)
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        try await request(path, method: "GET", body: nil)
    }

    private func request<T: Decodable>(_ path: String, method: String, body: Data?) async throws -> T {
        let data = try await responseData(path, method: method, body: body)
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw EngineError.server("Invalid engine response: \(error.localizedDescription)")
        }
    }

    private func responseData(_ path: String, method: String, body: Data?) async throws -> Data {
        guard let baseURL, let url = URL(string: path, relativeTo: baseURL) else {
            throw EngineError.invalidResponse
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = 30
        request.httpBody = body
        if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw EngineError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "HTTP \(http.statusCode)"
            throw EngineError.server(message)
        }
        return data
    }

    private struct Health: Decodable { let version: String }

    private func health(at url: URL) async throws -> Health {
        var request = URLRequest(url: url.appendingPathComponent("api/health"))
        request.timeoutInterval = 2
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw EngineError.invalidResponse
        }
        return try decoder.decode(Health.self, from: data)
    }

    private func locateRuntime() throws -> RuntimeLocations {
        let fm = FileManager.default
        var bundledAlgorithm: URL?
        if let resources = Bundle.main.resourceURL {
            let algorithm = resources.appendingPathComponent("Avril-V3.0-Algorithm", isDirectory: true)
            if isAlgorithmDirectory(algorithm) {
                bundledAlgorithm = algorithm
            }
            let runtime = resources.appendingPathComponent("python-runtime", isDirectory: true)
            let bundledPythonCandidates = [
                runtime.appendingPathComponent("bin/python3"),
                runtime.appendingPathComponent("bin/python3.13"),
            ]
            if let python = bundledPythonCandidates.first(where: { fm.isExecutableFile(atPath: $0.path) }),
               isAlgorithmDirectory(algorithm) {
                return RuntimeLocations(python: python, algorithm: algorithm, pythonHome: runtime)
            }
        }

        let environment = ProcessInfo.processInfo.environment
        var algorithmCandidates = [URL]()
        if let bundledAlgorithm {
            algorithmCandidates.append(bundledAlgorithm)
        }
        algorithmCandidates.append(contentsOf: [
                environment["AVRIL_NOVA_ENGINE_DIR"],
                environment["AVRIL_ENGINE_DIR"],
            ]
            .compactMap { $0 }
            .map { URL(fileURLWithPath: $0, isDirectory: true) })

        guard let algorithm = algorithmCandidates.first(where: isAlgorithmDirectory) else {
            throw EngineError.missingAlgorithm
        }

        let pythonCandidates = [
            environment["AVRIL_NOVA_PYTHON"],
            environment["AVRIL_PYTHON"],
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        .compactMap { $0 }
        .map { URL(fileURLWithPath: $0) }

        guard let python = pythonCandidates.first(where: { fm.isExecutableFile(atPath: $0.path) }) else {
            throw EngineError.missingRuntime
        }
        return RuntimeLocations(python: python, algorithm: algorithm, pythonHome: nil)
    }

    private func isAlgorithmDirectory(_ url: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: url.appendingPathComponent("desktop_bridge.py").path
        )
    }

    private func applicationSupportDirectory() throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let target = base.appendingPathComponent("Avril Nova", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        return target
    }
}
