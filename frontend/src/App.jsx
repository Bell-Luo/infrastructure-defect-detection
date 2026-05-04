import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'

// 修改为你的IP地址，让局域网内的队友也能访问
const API_BASE = 'http://10.98.245.42:8000/api'
const RESOURCE_BASE = 'http://10.98.245.42:8000'
axios.defaults.timeout = 300000

function App() {
  const [healthStatus, setHealthStatus] = useState(null)
  const [healthLoading, setHealthLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState(null)
  const [modelInfoLoading, setModelInfoLoading] = useState(false)
  const [datasetInfo, setDatasetInfo] = useState(null)
  const [datasetLoading, setDatasetLoading] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [detectionResult, setDetectionResult] = useState(null)
  const [detectionLoading, setDetectionLoading] = useState(false)
  const [viewMode, setViewMode] = useState('result')
  const [selectedVideo, setSelectedVideo] = useState(null)
  const [videoResult, setVideoResult] = useState(null)
  const [videoLoading, setVideoLoading] = useState(false)
  const [sampleInterval, setSampleInterval] = useState(5)
  const [originalImageSrc, setOriginalImageSrc] = useState('')
  const [analysisInfo, setAnalysisInfo] = useState(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const canvasRef = useRef(null)
  const imageRef = useRef(null)

  const checkHealth = async () => {
    setHealthLoading(true)
    try {
      const response = await axios.get(`${API_BASE}/health`)
      setHealthStatus(response.data)
    } catch (error) {
      console.error('健康检查失败:', error)
      setHealthStatus({ status: 'error', message: error.message || '无法连接到后端服务' })
    } finally {
      setHealthLoading(false)
    }
  }

  const fetchModelInfo = async () => {
    setModelInfoLoading(true)
    try {
      const response = await axios.get(`${API_BASE}/model/info`)
      setModelInfo(response.data)
    } catch (error) {
      console.error('获取模型信息失败:', error)
      setModelInfo({ status: 'error', message: error.message || '获取模型信息失败' })
    } finally {
      setModelInfoLoading(false)
    }
  }

  const fetchDatasetInfo = async () => {
    setDatasetLoading(true)
    try {
      const response = await axios.get(`${API_BASE}/dataset/info`)
      setDatasetInfo(response.data)
    } catch (error) {
      console.error('获取数据集信息失败:', error)
      setDatasetInfo({ status: 'error', message: error.message || '获取数据集信息失败' })
    } finally {
      setDatasetLoading(false)
    }
  }

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0])
      setDetectionResult(null)
      setOriginalImageSrc('')
    }
  }

  const resetImageSelection = () => {
    setSelectedFile(null)
    setDetectionResult(null)
    setOriginalImageSrc('')
    document.getElementById('image-input').value = ''
  }

  const handleImageDetect = async () => {
    if (!selectedFile) {
      alert('请先选择图片')
      return
    }

    setDetectionLoading(true)
    setDetectionResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await axios.post(`${API_BASE}/detect/image`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      setDetectionResult(response.data)
      setOriginalImageSrc(`${RESOURCE_BASE}${response.data.original_image_url}`)
    } catch (error) {
      console.error('检测失败:', error)
      alert('检测失败: ' + (error.response?.data?.message || error.message || '未知错误'))
    } finally {
      setDetectionLoading(false)
    }
  }

  const handleVideoChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedVideo(e.target.files[0])
      setVideoResult(null)
    }
  }

  const resetVideoSelection = () => {
    setSelectedVideo(null)
    setVideoResult(null)
    document.getElementById('video-input').value = ''
  }

  const handleVideoDetect = async () => {
    if (!selectedVideo) {
      alert('请先选择视频')
      return
    }

    setVideoLoading(true)
    setVideoResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedVideo)

      const response = await axios.post(`${API_BASE}/detect/video`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 300000,
      })

      setVideoResult(response.data)
    } catch (error) {
      console.error('视频检测失败:', error)
      alert('视频检测失败: ' + (error.response?.data?.message || error.message || '未知错误'))
    } finally {
      setVideoLoading(false)
    }
  }

  useEffect(() => {
    if (viewMode === 'canvas' && detectionResult && imageRef.current && canvasRef.current) {
      drawDetectionsOnCanvas()
    }
  }, [viewMode, detectionResult, originalImageSrc])

  const drawDetectionsOnCanvas = () => {
    const canvas = canvasRef.current
    const img = imageRef.current
    if (!canvas || !img) {
      console.log('Canvas或Image未加载')
      return
    }

    const ctx = canvas.getContext('2d')
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0)

    const colors = ['#4ECDC4', '#FF6B6B', '#FFE66D', '#95E1D3', '#FF8A80']

    detectionResult.detections.forEach((det, idx) => {
      const [x1, y1, x2, y2] = det.bbox
      const color = colors[det.class_id % colors.length]

      ctx.strokeStyle = color
      ctx.lineWidth = 3
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

      ctx.fillStyle = color
      ctx.globalAlpha = 0.9
      ctx.fillRect(x1, y1 - 25, 150, 22)

      ctx.fillStyle = '#FFFFFF'
      ctx.font = 'bold 13px system-ui, sans-serif'
      ctx.fillText(`${det.class_name} ${(det.confidence * 100).toFixed(1)}%`, x1 + 6, y1 - 7)
      ctx.globalAlpha = 1
    })
  }

  const downloadImage = () => {
    if (detectionResult && detectionResult.result_image_url) {
      const link = document.createElement('a')
      link.href = `${RESOURCE_BASE}${detectionResult.result_image_url}`
      link.download = 'detection_result.jpg'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const downloadCSV = () => {
    if (detectionResult && detectionResult.report_url) {
      const link = document.createElement('a')
      link.href = `${RESOURCE_BASE}${detectionResult.report_url}`
      link.download = 'detection_report.csv'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const downloadVideoReport = () => {
    if (videoResult && videoResult.report_url) {
      const link = document.createElement('a')
      link.href = `${RESOURCE_BASE}${videoResult.report_url}`
      link.download = 'video_detection_report.csv'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const generatePDFReport = async () => {
    try {
      const reportData = detectionResult || videoResult || null
      const response = await axios.post(`${API_BASE}/report/pdf`, reportData)
      if (response.data.status === 'success' && response.data.report_url) {
        const link = document.createElement('a')
        link.href = `${RESOURCE_BASE}${response.data.report_url}`
        link.download = 'detection_report.pdf'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      }
    } catch (error) {
      console.error('生成PDF失败:', error)
      alert('生成PDF失败，请确保已安装reportlab库')
    }
  }

  const fetchAnalysis = async () => {
    setAnalysisLoading(true)
    try {
      const response = await axios.get(`${API_BASE}/analysis/issues`)
      setAnalysisInfo(response.data)
    } catch (error) {
      console.error('获取分析失败:', error)
      setAnalysisInfo({ status: 'error', message: error.message || '获取分析失败' })
    } finally {
      setAnalysisLoading(false)
    }
  }

  const safeArray = (val) => Array.isArray(val) ? val : []
  const safeObject = (val) => (val && typeof val === 'object' && !Array.isArray(val)) ? val : {}
  const safeJoin = (arr, separator = ', ') => Array.isArray(arr) ? arr.join(separator) : String(arr || '')

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>基础设施外观缺陷智能检测系统</h1>
        </div>
      </header>

      <main className="main">
        <section className="card health-card">
          <h2>系统健康检查</h2>
          <button className="btn btn-primary" onClick={checkHealth} disabled={healthLoading}>
            {healthLoading ? '检查中...' : '检查后端健康状态'}
          </button>

          {healthStatus && (
            <div className="health-result">
              <div className="status-grid">
                <div className="status-item">
                  <span className="status-label">服务状态</span>
                  <span className={`status-value ${healthStatus.status === 'ok' ? 'success' : 'error'}`}>
                    {healthStatus.status === 'ok' ? '正常运行' : '服务异常'}
                  </span>
                </div>
                <div className="status-item">
                  <span className="status-label">健康检查时间</span>
                  <span className="status-value">{healthStatus.timestamp || new Date().toLocaleString()}</span>
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="card health-card">
          <h2>模型训练信息</h2>
          <button className="btn btn-primary" onClick={fetchModelInfo} disabled={modelInfoLoading}>
            {modelInfoLoading ? '加载中...' : '获取模型信息'}
          </button>

          {modelInfo && (
            <div className="health-result">
              <div className="status-grid">
                <div className="status-item">
                  <span className="status-label">模型加载状态</span>
                  <span className={`status-value ${modelInfo.model_loaded ? 'success' : 'warning'}`}>
                    {modelInfo.model_loaded ? '已加载' : '未加载（模拟模式）'}
                  </span>
                </div>
                <div className="status-item">
                  <span className="status-label">模型路径</span>
                  <span className="status-value">{modelInfo.model_path || 'N/A'}</span>
                </div>
                <div className="status-item">
                  <span className="status-label">模型加载时间</span>
                  <span className="status-value">{modelInfo.model_load_time?.toFixed(3) || 0} 秒</span>
                </div>
                <div className="status-item">
                  <span className="status-label">加载时间戳</span>
                  <span className="status-value">{modelInfo.model_load_timestamp || 'N/A'}</span>
                </div>
                <div className="status-item">
                  <span className="status-label">置信度阈值</span>
                  <span className="status-value">{modelInfo.confidence_threshold || 'N/A'}</span>
                </div>
                <div className="status-item">
                  <span className="status-label">检测类别</span>
                  <span className="status-value">{safeJoin(modelInfo.classes, ', ')}</span>
                </div>

                {modelInfo.training_stats && modelInfo.training_stats.samples_used > 0 && (
                  <>
                    <div className="status-divider">训练指标</div>
                    <div className="status-item">
                      <span className="status-label">训练样本数</span>
                      <span className="status-value">{modelInfo.training_stats.samples_used} 张图片</span>
                    </div>
                    <div className="status-item">
                      <span className="status-label">训练轮次</span>
                      <span className="status-value">{modelInfo.training_stats.epochs_trained} epochs</span>
                    </div>
                    <div className="status-item">
                      <span className="status-label">精确率 (Precision)</span>
                      <span className="status-value">{modelInfo.training_stats.precision > 0 ? (modelInfo.training_stats.precision * 100).toFixed(2) + '%' : 'N/A'}</span>
                    </div>
                    <div className="status-item">
                      <span className="status-label">召回率 (Recall)</span>
                      <span className="status-value">{modelInfo.training_stats.recall > 0 ? (modelInfo.training_stats.recall * 100).toFixed(2) + '%' : 'N/A'}</span>
                    </div>
                    <div className="status-item">
                      <span className="status-label">mAP@0.5</span>
                      <span className="status-value">{modelInfo.training_stats.mAP50 > 0 ? (modelInfo.training_stats.mAP50 * 100).toFixed(2) + '%' : 'N/A'}</span>
                    </div>
                    <div className="status-item">
                      <span className="status-label">mAP@0.5:0.95</span>
                      <span className="status-value">{modelInfo.training_stats.mAP50_95 > 0 ? (modelInfo.training_stats.mAP50_95 * 100).toFixed(2) + '%' : 'N/A'}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </section>

        <section className="card detection-card">
          <h2>图片缺陷检测</h2>

          <div className="upload-section">
            <input type="file" accept="image/*" onChange={handleImageChange} id="image-input" style={{ display: 'none' }} />
            <label htmlFor="image-input" className="btn btn-upload">
              {selectedFile ? `已选择: ${selectedFile.name}` : '选择图片'}
            </label>

            {selectedFile && (
              <div className="upload-actions">
                <button className="btn btn-primary" onClick={handleImageDetect} disabled={detectionLoading}>
                  {detectionLoading ? '检测中...' : '开始检测'}
                </button>
                <button className="btn btn-secondary" onClick={resetImageSelection}>
                  重新选择
                </button>
              </div>
            )}
          </div>

          {selectedFile && !detectionResult && (
            <div className="file-info">
              <span>文件大小: {(selectedFile.size / 1024).toFixed(1)} KB</span>
            </div>
          )}

          {detectionResult && detectionResult.status === 'success' && (
            <div className="result-section">
              <div className="result-header">
                <h3>检测完成</h3>
                <p>共检测到 {detectionResult.statistics?.total_detections || 0} 个缺陷</p>
                <div className="result-meta">
                  <span>检测时间: {detectionResult.statistics?.detection_time?.toFixed(3) || 0}秒</span>
                  <span>使用模型: {detectionResult.model_used || 'preloaded'}</span>
                </div>
              </div>

              <div className="view-toggle">
                <button className={`toggle-btn ${viewMode === 'original' ? 'active' : ''}`} onClick={() => setViewMode('original')}>原图</button>
                <button className={`toggle-btn ${viewMode === 'result' ? 'active' : ''}`} onClick={() => setViewMode('result')}>检测结果图</button>
                <button className={`toggle-btn ${viewMode === 'canvas' ? 'active' : ''}`} onClick={() => setViewMode('canvas')}>Canvas绘制</button>
              </div>

              <div className="image-display">
                {viewMode === 'canvas' ? (
                  <div className="canvas-container">
                    <img 
                      ref={imageRef} 
                      src={originalImageSrc} 
                      alt="原图" 
                      style={{ display: 'none' }} 
                      onLoad={() => {
                        console.log('原图加载完成，开始绘制')
                        drawDetectionsOnCanvas()
                      }}
                    />
                    <canvas ref={canvasRef} className="result-image" />
                  </div>
                ) : (
                  <img
                    src={viewMode === 'original' ? originalImageSrc : `${RESOURCE_BASE}${detectionResult.result_image_url}`}
                    alt={viewMode === 'original' ? '原图' : '检测结果图'}
                    className="result-image"
                  />
                )}
              </div>

              <div className="detections-list">
                <h4>检测详情</h4>
                <div className="detection-items">
                  {detectionResult.detections.map((det, idx) => (
                    <div key={idx} className="detection-item">
                      <span className="item-id">#{idx + 1}</span>
                      <span className="item-class">{det.class_name}</span>
                      <span className="item-conf">置信度: {(det.confidence * 100).toFixed(1)}%</span>
                      <span className="item-box">坐标: [{Math.round(det.bbox[0])}, {Math.round(det.bbox[1])}, {Math.round(det.bbox[2])}, {Math.round(det.bbox[3])}]</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="download-actions">
                <button className="btn btn-download" onClick={downloadImage}>保存结果图片</button>
                <button className="btn btn-download-report" onClick={downloadCSV}>导出CSV报告</button>
                <button className="btn btn-download-report" onClick={generatePDFReport}>导出PDF报告</button>
                <button className="btn btn-secondary" onClick={resetImageSelection}>上传新图片</button>
              </div>
            </div>
          )}
        </section>

        <section className="card detection-card">
          <h2>视频缺陷检测</h2>

          <div className="upload-section">
            <input type="file" accept="video/*" onChange={handleVideoChange} id="video-input" style={{ display: 'none' }} />
            <label htmlFor="video-input" className="btn btn-upload video">
              {selectedVideo ? `已选择: ${selectedVideo.name}` : '选择视频'}
            </label>

            {selectedVideo && (
              <div className="upload-actions">
                <button className="btn btn-primary" onClick={handleVideoDetect} disabled={videoLoading}>
                  {videoLoading ? '处理中...' : '开始检测'}
                </button>
                <button className="btn btn-secondary" onClick={resetVideoSelection}>
                  重新选择
                </button>
              </div>
            )}
          </div>

          {selectedVideo && !videoResult && (
            <div className="file-info">
              <span>文件大小: {(selectedVideo.size / (1024 * 1024)).toFixed(2)} MB</span>
              <span>支持格式: .mp4 .avi .mov</span>
            </div>
          )}

          {videoResult && videoResult.status === 'success' && (
            <div className="result-section">
              <div className="result-header">
                <h3>视频处理完成</h3>
              </div>

              <div className="info-group">
                <h4>视频信息</h4>
                <div className="video-info-grid">
                  <span>文件名: {videoResult.filename}</span>
                  <span>分辨率: {videoResult.video_info.width}x{videoResult.video_info.height}</span>
                  <span>帧率: {videoResult.video_info.fps} fps</span>
                  <span>总帧数: {videoResult.video_info.total_frames}</span>
                </div>
              </div>

              <div className="info-group">
                <h4>检测统计</h4>
                <div className="stats-grid">
                  <span>总检测数: {videoResult.statistics.total_detections}</span>
                  <span>处理时间: {videoResult.statistics.processing_time.toFixed(2)}秒</span>
                  <span>各类别统计:</span>
                  <ul className="class-stats">
                    {Object.entries(videoResult.statistics.class_statistics).map(([cls, count]) => (
                      <li key={cls}>{cls}: {count}个</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="info-group">
                <h4>结果视频</h4>
                <video controls className="result-video">
                  <source src={`${RESOURCE_BASE}${videoResult.result_video_url}`} type="video/mp4" />
                  您的浏览器不支持视频播放
                </video>
              </div>

              <div className="download-actions">
                <a href={`${RESOURCE_BASE}${videoResult.result_video_url}`} download className="btn btn-download">保存结果视频</a>
                <button className="btn btn-download-report" onClick={downloadVideoReport}>导出CSV报告</button>
                <button className="btn btn-download-report" onClick={generatePDFReport}>导出PDF报告</button>
                <button className="btn btn-secondary" onClick={resetVideoSelection}>上传新视频</button>
              </div>
            </div>
          )}
        </section>

        <section className="card dataset-card">
          <h2>数据集信息</h2>
          <button className="btn btn-primary" onClick={fetchDatasetInfo} disabled={datasetLoading}>
            {datasetLoading ? '加载中...' : '获取数据集信息'}
          </button>

          {datasetInfo && (
            <div className="dataset-result">
              {datasetInfo.status === 'error' ? (
                <div className="error-message">
                  <p>错误: {datasetInfo.message}</p>
                </div>
              ) : (
                <div className="dataset-details">
                  <div className="status-grid">
                    <div className="status-item">
                      <span className="status-label">数据集路径</span>
                      <span className={`status-value ${datasetInfo.path_exists ? 'success' : 'warning'}`}>
                        {datasetInfo.dataset_path || 'N/A'}
                      </span>
                    </div>
                    <div className="status-item">
                      <span className="status-label">路径状态</span>
                      <span className={`status-value ${datasetInfo.path_exists ? 'success' : 'warning'}`}>
                        {datasetInfo.path_exists ? '路径有效' : '路径不存在'}
                      </span>
                    </div>
                    <div className="status-item">
                      <span className="status-label">检测类别</span>
                      <span className="status-value">{safeJoin(datasetInfo.classes, ', ')}</span>
                    </div>
                    
                    {datasetInfo.statistics && (
                      <>
                        <div className="status-divider">数据集统计</div>
                        <div className="status-item">
                          <span className="status-label">训练集图片</span>
                          <span className="status-value">{datasetInfo.statistics.train_images} 张</span>
                        </div>
                        <div className="status-item">
                          <span className="status-label">验证集图片</span>
                          <span className="status-value">{datasetInfo.statistics.valid_images} 张</span>
                        </div>
                        <div className="status-item">
                          <span className="status-label">测试集图片</span>
                          <span className="status-value">{datasetInfo.statistics.test_images} 张</span>
                        </div>
                        <div className="status-item">
                          <span className="status-label">总图片数</span>
                          <span className="status-value">{datasetInfo.statistics.total_images} 张</span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        <section className="card health-card">
          <h2>误检漏检分析</h2>
          <button className="btn btn-primary" onClick={fetchAnalysis} disabled={analysisLoading}>
            {analysisLoading ? '加载中...' : '获取分析报告'}
          </button>

          {analysisInfo && (
            <div className="health-result">
              {analysisInfo.status === 'error' ? (
                <div className="error-message">
                  <p>错误: {analysisInfo.message}</p>
                </div>
              ) : (
                <>
                  {analysisInfo.overall_metrics && (
                    <>
                      <div className="status-divider">整体指标</div>
                      <div className="status-grid">
                        <div className="status-item">
                          <span className="status-label">精确率 (Precision)</span>
                          <span className="status-value">{analysisInfo.overall_metrics.precision > 0 ? (analysisInfo.overall_metrics.precision * 100).toFixed(2) + '%' : 'N/A'}</span>
                        </div>
                        <div className="status-item">
                          <span className="status-label">召回率 (Recall)</span>
                          <span className="status-value">{analysisInfo.overall_metrics.recall > 0 ? (analysisInfo.overall_metrics.recall * 100).toFixed(2) + '%' : 'N/A'}</span>
                        </div>
                        <div className="status-item">
                          <span className="status-label">mAP@0.5</span>
                          <span className="status-value">{analysisInfo.overall_metrics.mAP50 > 0 ? (analysisInfo.overall_metrics.mAP50 * 100).toFixed(2) + '%' : 'N/A'}</span>
                        </div>
                        <div className="status-item">
                          <span className="status-label">mAP@0.5:0.95</span>
                          <span className="status-value">{analysisInfo.overall_metrics.mAP50_95 > 0 ? (analysisInfo.overall_metrics.mAP50_95 * 100).toFixed(2) + '%' : 'N/A'}</span>
                        </div>
                      </div>
                    </>
                  )}
                  
                  {analysisInfo.issues && analysisInfo.issues.length > 0 && (
                    <>
                      <div className="status-divider">问题分析</div>
                      {analysisInfo.issues.map((issue, index) => (
                        <div key={index} className="status-item" style={{ width: '100%', display: 'block' }}>
                          <span className="status-label" style={{ color: issue.severity === 'high' ? '#FF6B6B' : issue.severity === 'medium' ? '#FFE66D' : '#4ECDC4' }}>
                            {issue.type}
                          </span>
                          <span className="status-value" style={{ fontSize: '0.9em', color: '#666' }}>
                            {issue.description}
                          </span>
                          {issue.suggestion && (
                            <span className="status-value" style={{ fontSize: '0.85em', color: '#4ECDC4', fontStyle: 'italic' }}>
                              建议: {issue.suggestion}
                            </span>
                          )}
                        </div>
                      ))}
                    </>
                  )}
                  
                  {analysisInfo.conclusion && (
                    <>
                      <div className="status-divider">总结</div>
                      <div className="status-item" style={{ width: '100%' }}>
                        <span className="status-value" style={{ fontSize: '0.95em' }}>{analysisInfo.conclusion}</span>
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
