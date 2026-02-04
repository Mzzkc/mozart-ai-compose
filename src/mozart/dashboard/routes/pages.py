"""Page routes for the Mozart Dashboard.

Handles HTML page rendering (non-API endpoints).
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from mozart.core.checkpoint import JobStatus
from mozart.dashboard.app import get_state_backend, get_templates
from mozart.state.base import StateBackend

router = APIRouter(tags=["Pages"])


@router.get("/", response_class=HTMLResponse)
async def dashboard_home(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    """Redirect to jobs list page."""
    return templates.TemplateResponse(
        "pages/jobs_list.html",
        {"request": request}
    )


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    """Render the jobs list page."""
    return templates.TemplateResponse(
        "pages/jobs_list.html",
        {"request": request}
    )


@router.get("/jobs/list", response_class=HTMLResponse)
async def jobs_list_partial(
    request: Request,
    status: JobStatus | None = Query(None, description="Filter by job status"),
    limit: int = Query(50, description="Maximum number of jobs"),
    templates: Jinja2Templates = Depends(get_templates),
    backend: StateBackend = Depends(get_state_backend),
) -> HTMLResponse:
    """Render the jobs list partial (HTMX target)."""
    try:
        # Fetch jobs from backend
        all_jobs = await backend.list_jobs()

        # Apply status filter if provided
        if status is not None:
            filtered_jobs = [j for j in all_jobs if j.status == status]
        else:
            filtered_jobs = all_jobs

        # Apply limit
        jobs = filtered_jobs[:limit]

        # Create job summaries for template
        from mozart.dashboard.routes import JobSummary
        job_summaries = [JobSummary.from_checkpoint(job) for job in jobs]

        return templates.TemplateResponse(
            "partials/jobs_list_content.html",
            {
                "request": request,
                "jobs": job_summaries,
                "total_jobs": len(all_jobs),
                "filtered_jobs": len(filtered_jobs),
                "applied_filter": status,
            }
        )
    except Exception as e:
        # Return error partial
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "error_title": "Failed to Load Jobs",
                "error_message": f"Unable to fetch job list: {str(e)}",
            }
        )


@router.get("/jobs/{job_id}/details", response_class=HTMLResponse)
async def job_details_page(
    job_id: str,
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    backend: StateBackend = Depends(get_state_backend),
) -> HTMLResponse:
    """Render job details page."""
    try:
        # Fetch job details
        state = await backend.load(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        from mozart.dashboard.routes import JobDetail
        job_detail = JobDetail.from_checkpoint(state)

        return templates.TemplateResponse(
            "pages/job_detail.html",
            {
                "request": request,
                "job": job_detail,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "error_title": "Failed to Load Job Details",
                "error_message": f"Unable to fetch job details: {str(e)}",
            }
        )


@router.get("/jobs/{job_id}/logs", response_class=HTMLResponse)
async def job_logs_page(
    job_id: str,
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    backend: StateBackend = Depends(get_state_backend),
) -> HTMLResponse:
    """Render job logs page."""
    try:
        # Fetch job details for context
        state = await backend.load(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        from mozart.dashboard.routes import JobDetail
        job_detail = JobDetail.from_checkpoint(state)

        return templates.TemplateResponse(
            "pages/job_logs.html",
            {
                "request": request,
                "job": job_detail,
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "error_title": "Failed to Load Job Logs",
                "error_message": f"Unable to fetch job logs: {str(e)}",
            }
        )


@router.get("/templates", response_class=HTMLResponse)
async def templates_page(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    """Render the templates browser page."""
    return templates.TemplateResponse(
        "pages/templates.html",
        {"request": request}
    )


@router.get("/editor", response_class=HTMLResponse)
async def score_editor_page(
    request: Request,
    template: str | None = None,
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    """Render the score editor page."""
    return templates.TemplateResponse(
        "pages/score_editor.html",
        {
            "request": request,
            "template_name": template,
        }
    )


@router.get("/api/templates/list", response_class=HTMLResponse)
async def templates_list_partial(
    request: Request,
    category: str | None = Query(None, description="Filter by category"),
    complexity: str | None = Query(None, description="Filter by complexity"),
    search: str | None = Query(None, description="Search templates"),
    templates: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    """Render the templates grid partial (HTMX target)."""
    try:
        # Import template analysis function from scores router
        import yaml
        import re
        from mozart.scores.templates import list_templates, get_template_path, TEMPLATE_FILES

        template_dict = list_templates()
        filtered_templates = []

        # Category mapping for analysis
        category_map = {
            'simple-task': 'workflow',
            'multi-sheet': 'workflow',
            'review-cycle': 'workflow',
            'data-processing': 'data',
            'testing-workflow': 'testing',
            'deployment-pipeline': 'deployment'
        }

        for name in template_dict:
            try:
                template_path = get_template_path(name)
                content = template_path.read_text()

                # Parse YAML to extract metadata
                data = yaml.safe_load(content)
                sheets = data.get('sheet', {}).get('total_sheets', 1)

                # Determine complexity
                if sheets == 1:
                    tmpl_complexity = 'simple'
                elif sheets <= 3:
                    tmpl_complexity = 'medium'
                else:
                    tmpl_complexity = 'complex'

                tmpl_category = category_map.get(name, 'general')

                # Apply filters
                if category and tmpl_category != category:
                    continue
                if complexity and tmpl_complexity != complexity:
                    continue
                if search and search.lower() not in name.lower() and search.lower() not in data.get('name', '').lower():
                    continue

                # Extract features
                features = []
                if 'dependencies' in data.get('sheet', {}):
                    features.append('Multi-sheet dependencies')
                if data.get('validations'):
                    features.append(f"{len(data['validations'])} validation checks")
                if data.get('notifications'):
                    features.append('Automated notifications')
                if '{{' in content:
                    features.append('Customizable variables')

                # Extract variables
                variables = []
                var_pattern = r'{{\s*(\w+)(?:\s*\|\s*default\([^)]*\))?\s*}}'
                var_matches = re.findall(var_pattern, content)
                for var in set(var_matches):
                    variables.append({
                        'name': var,
                        'description': f'Template variable: {var}'
                    })

                template = {
                    'name': name,
                    'title': name.replace('-', ' ').title(),
                    'filename': TEMPLATE_FILES[name],
                    'description': f"A {tmpl_complexity} {tmpl_category} template with {sheets} sheet{'s' if sheets != 1 else ''}",
                    'complexity': tmpl_complexity,
                    'sheets': sheets,
                    'category': tmpl_category,
                    'features': features,
                    'variables': variables,
                    'content': content,
                    'estimated_duration': f"{sheets * 15}-{sheets * 30} min" if sheets > 1 else "5-15 min"
                }

                filtered_templates.append(template)

            except Exception:
                # Skip templates that can't be loaded
                continue

        return templates.TemplateResponse(
            "partials/templates_grid.html",
            {
                "request": request,
                "templates": filtered_templates,
                "search": search,
                "category": category,
                "complexity": complexity,
            }
        )

    except Exception as e:
        # Return error partial
        return templates.TemplateResponse(
            "partials/error_message.html",
            {
                "request": request,
                "error_title": "Failed to Load Templates",
                "error_message": f"Unable to fetch template list: {str(e)}",
            }
        )
