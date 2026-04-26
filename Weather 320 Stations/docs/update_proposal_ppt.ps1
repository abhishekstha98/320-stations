Add-Type -AssemblyName System.IO.Compression.FileSystem

$ErrorActionPreference = "Stop"

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pptPath = Join-Path $baseDir "Proposal Defense.pptx"
$backupPath = Join-Path $baseDir "Proposal Defense.before_update.bak.pptx"
$tempDir = Join-Path $env:TEMP ("proposal_ppt_update_" + [guid]::NewGuid())

function New-Paragraph {
    param(
        [xml]$Doc,
        [System.Xml.XmlNamespaceManager]$Ns,
        [string]$Text,
        [int]$Level = -1,
        [switch]$NoBullet,
        [string]$Align
    )

    $p = $Doc.CreateElement("a", "p", $Ns.LookupNamespace("a"))
    if ($Level -ge 0 -or $NoBullet -or $Align) {
        $pPr = $Doc.CreateElement("a", "pPr", $Ns.LookupNamespace("a"))
        if ($Level -ge 0) {
            $lvlAttr = $Doc.CreateAttribute("lvl")
            $lvlAttr.Value = [string]$Level
            [void]$pPr.Attributes.Append($lvlAttr)
        }
        if ($Align) {
            $algnAttr = $Doc.CreateAttribute("algn")
            $algnAttr.Value = $Align
            [void]$pPr.Attributes.Append($algnAttr)
        }
        if ($NoBullet) {
            $buNone = $Doc.CreateElement("a", "buNone", $Ns.LookupNamespace("a"))
            [void]$pPr.AppendChild($buNone)
        }
        [void]$p.AppendChild($pPr)
    }

    $r = $Doc.CreateElement("a", "r", $Ns.LookupNamespace("a"))
    $rPr = $Doc.CreateElement("a", "rPr", $Ns.LookupNamespace("a"))
    $langAttr = $Doc.CreateAttribute("lang")
    $langAttr.Value = "en-US"
    [void]$rPr.Attributes.Append($langAttr)
    [void]$r.AppendChild($rPr)

    $t = $Doc.CreateElement("a", "t", $Ns.LookupNamespace("a"))
    $t.InnerText = $Text
    [void]$r.AppendChild($t)
    [void]$p.AppendChild($r)

    $end = $Doc.CreateElement("a", "endParaRPr", $Ns.LookupNamespace("a"))
    $endLang = $Doc.CreateAttribute("lang")
    $endLang.Value = "en-US"
    [void]$end.Attributes.Append($endLang)
    [void]$p.AppendChild($end)

    return $p
}

function Set-ShapeParagraphs {
    param(
        [xml]$Doc,
        [System.Xml.XmlNamespaceManager]$Ns,
        [string]$ShapeId,
        [array]$Paragraphs
    )

    $shape = $Doc.SelectSingleNode("//p:sp[p:nvSpPr/p:cNvPr[@id='$ShapeId']]", $Ns)
    if (-not $shape) {
        throw "Shape id $ShapeId not found."
    }

    $txBody = $shape.SelectSingleNode("./p:txBody", $Ns)
    if (-not $txBody) {
        throw "Text body for shape id $ShapeId not found."
    }

    $bodyPr = $txBody.SelectSingleNode("./a:bodyPr", $Ns).CloneNode($true)
    $lstStyleNode = $txBody.SelectSingleNode("./a:lstStyle", $Ns)
    if ($lstStyleNode) {
        $lstStyle = $lstStyleNode.CloneNode($true)
    } else {
        $lstStyle = $Doc.CreateElement("a", "lstStyle", $Ns.LookupNamespace("a"))
    }

    $txBody.RemoveAll()
    [void]$txBody.AppendChild($bodyPr)
    [void]$txBody.AppendChild($lstStyle)

    foreach ($para in $Paragraphs) {
        [void]$txBody.AppendChild((New-Paragraph -Doc $Doc -Ns $Ns -Text $para.text -Level $para.level -NoBullet:([bool]$para.noBullet) -Align $para.align))
    }
}

function Remove-MatchingNodes {
    param(
        [xml]$Doc,
        [System.Xml.XmlNamespaceManager]$Ns,
        [string]$XPath
    )

    $nodes = @($Doc.SelectNodes($XPath, $Ns))
    foreach ($node in $nodes) {
        [void]$node.ParentNode.RemoveChild($node)
    }
}

New-Item -ItemType Directory -Path $tempDir | Out-Null
[System.IO.Compression.ZipFile]::ExtractToDirectory($pptPath, $tempDir)

$slideUpdates = @{
    "slide3.xml" = @{
        ShapeId = "3"
        Paragraphs = @(
            @{ text = "Recent foundation: Bhandari et al. (2024) show GNN relevance for Nepal weather prediction in data-scarce environments."; level = -1; noBullet = $false; align = $null },
            @{ text = "Agriculture in Nepal is highly sensitive to short-range weather variability."; level = -1; noBullet = $false; align = $null },
            @{ text = "Complex terrain creates sharp microclimates across short distances."; level = -1; noBullet = $false; align = $null },
            @{ text = "Distance-based neighborhood assumptions often fail across ridges, valleys, and elevation gradients."; level = -1; noBullet = $false; align = $null },
            @{ text = "The core idea is to connect stations by meteorological affinity, not map distance alone."; level = -1; noBullet = $false; align = $null }
        )
    }
    "slide5.xml" = @{
        ShapeId = "3"
        Paragraphs = @(
            @{ text = "Distance-based adjacency assumes spatial smoothness."; level = -1; noBullet = $false; align = $null },
            @{ text = "That assumption breaks in mountains because terrain barriers and elevation contrasts distort local weather."; level = -1; noBullet = $false; align = $null },
            @{ text = "Two nearby stations can behave very differently, while distant valleys can show similar signal patterns."; level = -1; noBullet = $false; align = $null },
            @{ text = "A misleading graph causes spatial aggregation to mix useful information with noise."; level = -1; noBullet = $false; align = $null },
            @{ text = "Result: weaker forecasts and less reliable downstream agricultural support."; level = -1; noBullet = $false; align = $null }
        )
    }
    "slide6.xml" = @{
        ShapeId = "3"
        Paragraphs = @(
            @{ text = "Bhandari et al. (2024) provide the recent Nepal-specific graph-weather foundation."; level = -1; noBullet = $false; align = $null },
            @{ text = "Their work supports GNNs for weather prediction/imputation in data-scarce complex terrain."; level = -1; noBullet = $false; align = $null },
            @{ text = "Complex-terrain weather forecasting needs a graph that reflects functional similarity between stations."; level = -1; noBullet = $false; align = $null },
            @{ text = "A matched comparison between correlation-based and distance-based STGCN graphs is still needed."; level = -1; noBullet = $false; align = $null },
            @{ text = "Agricultural advisory systems also need a stronger, transparent forecasting backbone."; level = -1; noBullet = $false; align = $null }
        )
    }
    "slide7.xml" = @{
        ShapeId = "3"
        Paragraphs = @(
            @{ text = "General Objective"; level = -1; noBullet = $true; align = $null },
            @{ text = "Design and evaluate a weather forecasting framework for complex terrain using correlation-defined station connectivity."; level = 1; noBullet = $false; align = $null },
            @{ text = "Specific Objectives"; level = -1; noBullet = $true; align = $null },
            @{ text = "Construct a correlation-adaptive graph for 320 stations using absolute Pearson correlation and top-33% sparsification."; level = 1; noBullet = $false; align = $null },
            @{ text = "Prepare 43-lag inputs and a 7-step forecast horizon from multivariate daily weather records."; level = 1; noBullet = $false; align = $null },
            @{ text = "Implement a three-block STGCN to predict T2M_MIN, RH2M, and PRECTOTCORR."; level = 1; noBullet = $false; align = $null },
            @{ text = "Compare the correlation graph with a distance-based graph under matched settings."; level = 1; noBullet = $false; align = $null },
            @{ text = "Position the forecasting outputs for future crop advisory integration."; level = 1; noBullet = $false; align = $null }
        )
    }
    "slide10.xml" = @{
        ShapeId = "5"
        Paragraphs = @(
            @{ text = "Input"; level = -1; noBullet = $true; align = $null },
            @{ text = "320 stations x 43 lags x 14 weather features"; level = 1; noBullet = $false; align = $null },
            @{ text = "Architecture"; level = -1; noBullet = $true; align = $null },
            @{ text = "Three ST-Conv blocks with temporal kernels 9, 7, and 5"; level = 1; noBullet = $false; align = $null },
            @{ text = "Chebyshev graph convolution orders 4, 4, and 3"; level = 1; noBullet = $false; align = $null },
            @{ text = "Output"; level = -1; noBullet = $true; align = $null },
            @{ text = "Seven-step forecasts for T2M_MIN, RH2M, and PRECTOTCORR"; level = 1; noBullet = $false; align = $null },
            @{ text = "Key Idea"; level = -1; noBullet = $true; align = $null },
            @{ text = "Learn temporal patterns while propagating information across correlated stations"; level = 1; noBullet = $false; align = $null }
        )
    }
    "slide11.xml" = @{
        TitleShapeId = "2"
        Title = "Methodology (Application Relevance and Future Integration)"
        ShapeId = "5"
        Paragraphs = @(
            @{ text = "Input"; level = -1; noBullet = $true; align = $null },
            @{ text = "Seven-day forecasts of minimum temperature, relative humidity, and precipitation"; level = 1; noBullet = $false; align = $null },
            @{ text = "Immediate Use"; level = -1; noBullet = $true; align = $null },
            @{ text = "Nearest-station weather outlook for a user-supplied location"; level = 1; noBullet = $false; align = $null },
            @{ text = "Near-Term Extension"; level = -1; noBullet = $true; align = $null },
            @{ text = "Map forecasts into frost-risk, humidity-regime, and rainfall classes"; level = 1; noBullet = $false; align = $null },
            @{ text = "Future Integration"; level = -1; noBullet = $true; align = $null },
            @{ text = "Crop suitability rules or advisory models can consume the forecast backbone"; level = 1; noBullet = $false; align = $null },
            @{ text = "Scope Note"; level = -1; noBullet = $true; align = $null },
            @{ text = "Crop recommendation is downstream motivation, not a completed evaluated module in the current evidence base"; level = 1; noBullet = $false; align = $null }
        )
    }
    "slide12.xml" = @{
        ShapeId = "3"
        Paragraphs = @(
            @{ text = "Primary Comparison"; level = -1; noBullet = $true; align = $null },
            @{ text = "Correlation-adaptive STGCN vs distance-based STGCN"; level = 1; noBullet = $false; align = $null },
            @{ text = "Controlled Factors"; level = -1; noBullet = $true; align = $null },
            @{ text = "Same station set, features, forecast horizon, and model family"; level = 1; noBullet = $false; align = $null },
            @{ text = "Evaluation Protocol"; level = -1; noBullet = $true; align = $null },
            @{ text = "Chronological holdout as the main test condition"; level = 1; noBullet = $false; align = $null },
            @{ text = "Random split as a secondary reference for leakage awareness"; level = 1; noBullet = $false; align = $null },
            @{ text = "Reported Outputs"; level = -1; noBullet = $true; align = $null },
            @{ text = "Per-variable, per-horizon metrics plus convergence and physical plausibility checks"; level = 1; noBullet = $false; align = $null },
            @{ text = "Literature Benchmark"; level = -1; noBullet = $true; align = $null },
            @{ text = "Bhandari et al. (2024) is cited as the recent Nepal-specific benchmark, not the direct baseline"; level = 1; noBullet = $false; align = $null }
        )
    }
    "slide16.xml" = @{
        ShapeId = "3"
        Paragraphs = @(
            @{ text = "Methodological"; level = -1; noBullet = $true; align = $null },
            @{ text = "A transparent correlation-based graph construction strategy for complex terrain"; level = -1; noBullet = $false; align = $null },
            @{ text = "Technical"; level = -1; noBullet = $true; align = $null },
            @{ text = "A convergent 320-station STGCN forecasting backbone for T2M_MIN, RH2M, and PRECTOTCORR"; level = -1; noBullet = $false; align = $null },
            @{ text = "Analytical"; level = -1; noBullet = $true; align = $null },
            @{ text = "A clearer comparison between correlation-defined and distance-defined adjacency"; level = -1; noBullet = $false; align = $null },
            @{ text = "Literature"; level = -1; noBullet = $true; align = $null },
            @{ text = "Problem statement anchored in Bhandari et al. (2024), a recent Nepal-focused GNN weather paper"; level = -1; noBullet = $false; align = $null },
            @{ text = "Applied"; level = -1; noBullet = $true; align = $null },
            @{ text = "Reusable forecasts that can support future agricultural decision support"; level = -1; noBullet = $false; align = $null },
            @{ text = "Thesis Position"; level = -1; noBullet = $true; align = $null },
            @{ text = "The validated contribution is the forecasting backbone, with crop recommendation framed as future work"; level = -1; noBullet = $false; align = $null }
        )
    }
    "slide17.xml" = @{
        TitleShapeId = "2"
        Title = "Timeline"
        ShapeId = "3"
        Paragraphs = @(
            @{ text = "Months 1-2: Data audit, preprocessing, and graph construction"; level = -1; noBullet = $false; align = $null },
            @{ text = "Month 3: Literature benchmarking with Bhandari et al. (2024) and distance-based baseline implementation"; level = -1; noBullet = $false; align = $null },
            @{ text = "Months 4-5: Model training, tuning, and convergence analysis"; level = -1; noBullet = $false; align = $null },
            @{ text = "Month 6: Chronological testing and comparative evaluation"; level = -1; noBullet = $false; align = $null },
            @{ text = "Month 7: Results writing, figures, limitations, and future integration discussion"; level = -1; noBullet = $false; align = $null },
            @{ text = "Month 8: Final thesis revision and defense preparation"; level = -1; noBullet = $false; align = $null }
        )
    }
}

foreach ($slideName in $slideUpdates.Keys) {
    $slidePath = Join-Path $tempDir ("ppt\slides\" + $slideName)
    [xml]$doc = Get-Content $slidePath
    $ns = New-Object System.Xml.XmlNamespaceManager($doc.NameTable)
    $ns.AddNamespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
    $ns.AddNamespace("p", "http://schemas.openxmlformats.org/presentationml/2006/main")
    $ns.AddNamespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")

    $update = $slideUpdates[$slideName]

    if ($update.ContainsKey("Title")) {
        Set-ShapeParagraphs -Doc $doc -Ns $ns -ShapeId $update.TitleShapeId -Paragraphs @(
            @{ text = $update.Title; level = -1; noBullet = $true; align = $null }
        )
    }

    Set-ShapeParagraphs -Doc $doc -Ns $ns -ShapeId $update.ShapeId -Paragraphs $update.Paragraphs
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($slidePath, $doc.OuterXml, $utf8NoBom)
}

foreach ($slideName in "slide13.xml", "slide14.xml", "slide15.xml") {
    $slidePath = Join-Path $tempDir ("ppt\slides\" + $slideName)
    [xml]$doc = Get-Content $slidePath
    $ns = New-Object System.Xml.XmlNamespaceManager($doc.NameTable)
    $ns.AddNamespace("a", "http://schemas.openxmlformats.org/drawingml/2006/main")
    $ns.AddNamespace("p", "http://schemas.openxmlformats.org/presentationml/2006/main")
    $ns.AddNamespace("mc", "http://schemas.openxmlformats.org/markup-compatibility/2006")

    Remove-MatchingNodes -Doc $doc -Ns $ns -XPath "//mc:AlternateContent"
    Remove-MatchingNodes -Doc $doc -Ns $ns -XPath "//p:graphicFrame"

    switch ($slideName) {
        "slide13.xml" {
            Set-ShapeParagraphs -Doc $doc -Ns $ns -ShapeId "2" -Paragraphs @(
                @{ text = "Evaluation Strategy"; level = -1; noBullet = $true; align = $null }
            )
            $spTree = $doc.SelectSingleNode("//p:spTree", $ns)
            $newShapeXml = @"
<p:sp xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:nvSpPr>
    <p:cNvPr id="20" name="TextBox 20"/>
    <p:cNvSpPr txBox="1"/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="1451579" y="1816441"/><a:ext cx="10293381" cy="4344428"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
  </p:spPr>
  <p:txBody>
    <a:bodyPr><a:normAutofit/></a:bodyPr>
    <a:lstStyle/>
  </p:txBody>
</p:sp>
"@
            $newShape = $doc.CreateDocumentFragment()
            $newShape.InnerXml = $newShapeXml
            [void]$spTree.AppendChild($newShape)
            Set-ShapeParagraphs -Doc $doc -Ns $ns -ShapeId "20" -Paragraphs @(
                @{ text = "Monitor training and validation loss throughout model fitting."; level = -1; noBullet = $false; align = $null },
                @{ text = "Report MSE, MAE, and RMSE for T2M_MIN, RH2M, and PRECTOTCORR."; level = -1; noBullet = $false; align = $null },
                @{ text = "Compute metrics separately for each of the 7 forecast steps."; level = -1; noBullet = $false; align = $null },
                @{ text = "Assess wet-day versus dry-day occurrence skill for precipitation."; level = -1; noBullet = $false; align = $null },
                @{ text = "Inspect whether forecast trajectories remain physically plausible for season and region."; level = -1; noBullet = $false; align = $null }
            )
        }
        "slide14.xml" {
            Set-ShapeParagraphs -Doc $doc -Ns $ns -ShapeId "2" -Paragraphs @(
                @{ text = "Validation and Verification"; level = -1; noBullet = $true; align = $null }
            )
            $spTree = $doc.SelectSingleNode("//p:spTree", $ns)
            $newShapeXml = @"
<p:sp xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:nvSpPr>
    <p:cNvPr id="20" name="TextBox 20"/>
    <p:cNvSpPr txBox="1"/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="1451579" y="1816441"/><a:ext cx="10293381" cy="4344428"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
  </p:spPr>
  <p:txBody>
    <a:bodyPr><a:normAutofit/></a:bodyPr>
    <a:lstStyle/>
  </p:txBody>
</p:sp>
"@
            $newShape = $doc.CreateDocumentFragment()
            $newShape.InnerXml = $newShapeXml
            [void]$spTree.AppendChild($newShape)
            Set-ShapeParagraphs -Doc $doc -Ns $ns -ShapeId "20" -Paragraphs @(
                @{ text = "Use a held-out test set for final forecasting assessment."; level = -1; noBullet = $false; align = $null },
                @{ text = "Run a matched baseline comparison to isolate the effect of graph construction."; level = -1; noBullet = $false; align = $null },
                @{ text = "Compare chronological and random split behavior to reveal possible optimistic validation."; level = -1; noBullet = $false; align = $null },
                @{ text = "Demonstrate inference through nearest-station forecasting for a query location."; level = -1; noBullet = $false; align = $null },
                @{ text = "Document reproducibility constraints such as hard-coded paths and missing intermediate artifacts."; level = -1; noBullet = $false; align = $null },
                @{ text = "Keep conclusions limited to supported implementation and evaluation evidence."; level = -1; noBullet = $false; align = $null }
            )
        }
        "slide15.xml" {
            Set-ShapeParagraphs -Doc $doc -Ns $ns -ShapeId "2" -Paragraphs @(
                @{ text = "Current Evidence Base"; level = -1; noBullet = $true; align = $null }
            )
            $spTree = $doc.SelectSingleNode("//p:spTree", $ns)
            $newShapeXml = @"
<p:sp xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:nvSpPr>
    <p:cNvPr id="20" name="TextBox 20"/>
    <p:cNvSpPr txBox="1"/>
    <p:nvPr/>
  </p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="1451579" y="1816441"/><a:ext cx="10293381" cy="4344428"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
  </p:spPr>
  <p:txBody>
    <a:bodyPr><a:normAutofit/></a:bodyPr>
    <a:lstStyle/>
  </p:txBody>
</p:sp>
"@
            $newShape = $doc.CreateDocumentFragment()
            $newShape.InnerXml = $newShapeXml
            [void]$spTree.AppendChild($newShape)
            Set-ShapeParagraphs -Doc $doc -Ns $ns -ShapeId "20" -Paragraphs @(
                @{ text = "Training logs show strong convergence over the recorded run."; level = -1; noBullet = $false; align = $null },
                @{ text = "The best recorded validation loss is about 0.0246 at printed epoch 171."; level = -1; noBullet = $false; align = $null },
                @{ text = "Example 7-step forecasts are available for both best-weight and last-weight checkpoints."; level = -1; noBullet = $false; align = $null },
                @{ text = "The repository clearly supports the forecasting backbone and inference workflow."; level = -1; noBullet = $false; align = $null },
                @{ text = "A complete matched baseline table is not yet available."; level = -1; noBullet = $false; align = $null },
                @{ text = "An end-to-end crop recommendation module is not yet implemented or validated."; level = -1; noBullet = $false; align = $null }
            )
        }
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($slidePath, $doc.OuterXml, $utf8NoBom)
}

if (-not (Test-Path $backupPath)) {
    Copy-Item $pptPath $backupPath
}

$newPptPath = Join-Path $baseDir "Proposal Defense.updated.pptx"
if (Test-Path $newPptPath) {
    Remove-Item $newPptPath -Force
}
[System.IO.Compression.ZipFile]::CreateFromDirectory($tempDir, $newPptPath)
Copy-Item $newPptPath $pptPath -Force
Remove-Item $tempDir -Recurse -Force

Write-Output "Updated deck saved to:"
Write-Output $pptPath
Write-Output "Backup saved to:"
Write-Output $backupPath
