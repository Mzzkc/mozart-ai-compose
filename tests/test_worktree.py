"""Tests for mozart.isolation.worktree module.

These tests verify the GitWorktreeManager implementation for creating,
managing, and cleaning up git worktrees for parallel job isolation.

Note: These tests require a git repository and actually create/remove
worktrees. They use temporary directories to avoid affecting the real repo.
"""

import subprocess
from pathlib import Path

import pytest

from mozart.isolation.worktree import (
    BranchExistsError,
    GitWorktreeManager,
    NotGitRepositoryError,
    WorktreeCreationError,
    WorktreeError,
    WorktreeInfo,
    WorktreeLockError,
    WorktreeRemovalError,
    WorktreeResult,
)


# --- Fixtures ---


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository for testing.

    Returns:
        Path to the temporary repository root.
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create initial commit (required for worktrees)
    readme = repo_path / "README.md"
    readme.write_text("# Test Repository\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return repo_path


@pytest.fixture
def manager(temp_git_repo: Path) -> GitWorktreeManager:
    """Create a GitWorktreeManager for the temp repo."""
    return GitWorktreeManager(temp_git_repo)


@pytest.fixture
def non_git_path(tmp_path: Path) -> Path:
    """Create a non-git directory."""
    path = tmp_path / "not_a_repo"
    path.mkdir()
    return path


# --- WorktreeInfo Tests ---


class TestWorktreeInfo:
    """Tests for WorktreeInfo dataclass."""

    def test_creation(self, tmp_path: Path) -> None:
        """Test WorktreeInfo can be created with required fields."""
        info = WorktreeInfo(
            path=tmp_path / "worktree",
            branch="mozart/test-job",
            commit="abc1234",
            locked=True,
            job_id="test-job",
        )
        assert info.path == tmp_path / "worktree"
        assert info.branch == "mozart/test-job"
        assert info.commit == "abc1234"
        assert info.locked is True
        assert info.job_id == "test-job"


class TestWorktreeResult:
    """Tests for WorktreeResult dataclass."""

    def test_success_result(self, tmp_path: Path) -> None:
        """Test successful result creation."""
        info = WorktreeInfo(
            path=tmp_path / "wt",
            branch="b",
            commit="c",
            locked=False,
            job_id="j",
        )
        result = WorktreeResult(success=True, worktree=info, error=None)
        assert result.success is True
        assert result.worktree == info
        assert result.error is None

    def test_failure_result(self) -> None:
        """Test failure result creation."""
        result = WorktreeResult(success=False, worktree=None, error="Something went wrong")
        assert result.success is False
        assert result.worktree is None
        assert result.error == "Something went wrong"


# --- Exception Tests ---


class TestExceptions:
    """Tests for worktree exception classes."""

    def test_worktree_error_is_exception(self) -> None:
        """Test base exception."""
        with pytest.raises(WorktreeError):
            raise WorktreeError("test error")

    def test_creation_error_inherits(self) -> None:
        """Test WorktreeCreationError inherits from WorktreeError."""
        with pytest.raises(WorktreeError):
            raise WorktreeCreationError("creation failed")

    def test_removal_error_inherits(self) -> None:
        """Test WorktreeRemovalError inherits from WorktreeError."""
        with pytest.raises(WorktreeError):
            raise WorktreeRemovalError("removal failed")

    def test_lock_error_inherits(self) -> None:
        """Test WorktreeLockError inherits from WorktreeError."""
        with pytest.raises(WorktreeError):
            raise WorktreeLockError("lock failed")

    def test_branch_exists_error_inherits(self) -> None:
        """Test BranchExistsError inherits from WorktreeError."""
        with pytest.raises(WorktreeError):
            raise BranchExistsError("branch exists")

    def test_not_git_repo_error_inherits(self) -> None:
        """Test NotGitRepositoryError inherits from WorktreeError."""
        with pytest.raises(WorktreeError):
            raise NotGitRepositoryError("not a repo")


# --- GitWorktreeManager Initialization Tests ---


class TestGitWorktreeManagerInit:
    """Tests for GitWorktreeManager initialization."""

    def test_initialization(self, temp_git_repo: Path) -> None:
        """Test manager can be initialized with a repo path."""
        manager = GitWorktreeManager(temp_git_repo)
        assert manager._repo_path == temp_git_repo.resolve()
        assert manager._git_verified is False

    def test_is_git_repository_true(self, temp_git_repo: Path) -> None:
        """Test is_git_repository returns True for git repos."""
        manager = GitWorktreeManager(temp_git_repo)
        assert manager.is_git_repository() is True

    def test_is_git_repository_false(self, non_git_path: Path) -> None:
        """Test is_git_repository returns False for non-git dirs."""
        manager = GitWorktreeManager(non_git_path)
        assert manager.is_git_repository() is False


# --- Create Worktree Tests ---


class TestCreateWorktree:
    """Tests for GitWorktreeManager.create_worktree()."""

    @pytest.mark.asyncio
    async def test_create_basic(self, manager: GitWorktreeManager, temp_git_repo: Path) -> None:
        """Test basic worktree creation."""
        result = await manager.create_worktree(
            job_id="test-job",
            lock=False,
        )

        assert result.success is True
        assert result.worktree is not None
        assert result.error is None

        # Verify worktree was created
        assert result.worktree.path.exists()
        assert result.worktree.branch == "mozart/test-job"
        assert result.worktree.job_id == "test-job"

    @pytest.mark.asyncio
    async def test_create_with_custom_prefix(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test worktree creation with custom branch prefix."""
        result = await manager.create_worktree(
            job_id="job-123",
            branch_prefix="custom",
            lock=False,
        )

        assert result.success is True
        assert result.worktree is not None
        assert result.worktree.branch == "custom/job-123"

    @pytest.mark.asyncio
    async def test_create_with_custom_base(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test worktree creation with custom base directory."""
        custom_base = temp_git_repo / "my_worktrees"
        result = await manager.create_worktree(
            job_id="job-custom",
            worktree_base=custom_base,
            lock=False,
        )

        assert result.success is True
        assert result.worktree is not None
        assert result.worktree.path.parent == custom_base

    @pytest.mark.asyncio
    async def test_create_with_lock(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test worktree creation with locking enabled."""
        result = await manager.create_worktree(
            job_id="locked-job",
            lock=True,
        )

        assert result.success is True
        assert result.worktree is not None
        assert result.worktree.locked is True

    @pytest.mark.asyncio
    async def test_create_not_git_repo(self, non_git_path: Path) -> None:
        """Test creation fails for non-git directory."""
        manager = GitWorktreeManager(non_git_path)

        with pytest.raises(NotGitRepositoryError):
            await manager.create_worktree(job_id="test")

    @pytest.mark.asyncio
    async def test_create_branch_exists(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test creation fails when branch already exists."""
        # Create a branch first
        subprocess.run(
            ["git", "branch", "mozart/existing-job"],
            cwd=temp_git_repo,
            check=True,
            capture_output=True,
        )

        with pytest.raises(BranchExistsError):
            await manager.create_worktree(job_id="existing-job")

    @pytest.mark.asyncio
    async def test_create_path_exists(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test creation fails when path already exists."""
        # Create the path first
        worktree_path = temp_git_repo / ".worktrees" / "blocked-job"
        worktree_path.mkdir(parents=True)

        with pytest.raises(WorktreeCreationError):
            await manager.create_worktree(job_id="blocked-job")


# --- Remove Worktree Tests ---


class TestRemoveWorktree:
    """Tests for GitWorktreeManager.remove_worktree()."""

    @pytest.mark.asyncio
    async def test_remove_existing(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test removing an existing worktree."""
        # Create first
        create_result = await manager.create_worktree(job_id="to-remove", lock=False)
        assert create_result.success
        assert create_result.worktree is not None
        worktree_path = create_result.worktree.path

        # Then remove
        remove_result = await manager.remove_worktree(worktree_path)
        assert remove_result.success is True
        assert not worktree_path.exists()

    @pytest.mark.asyncio
    async def test_remove_nonexistent_idempotent(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test removing nonexistent worktree succeeds (idempotent)."""
        nonexistent = temp_git_repo / "does_not_exist"

        result = await manager.remove_worktree(nonexistent)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_remove_with_force(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test removing worktree with force option."""
        create_result = await manager.create_worktree(job_id="force-remove", lock=False)
        assert create_result.worktree is not None
        worktree_path = create_result.worktree.path

        # Make worktree dirty
        dirty_file = worktree_path / "dirty.txt"
        dirty_file.write_text("uncommitted changes")

        # Should still succeed with force
        remove_result = await manager.remove_worktree(worktree_path, force=True)
        assert remove_result.success is True

    @pytest.mark.asyncio
    async def test_remove_locked_worktree(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test removing a locked worktree (should unlock first)."""
        create_result = await manager.create_worktree(job_id="locked-remove", lock=True)
        assert create_result.worktree is not None
        worktree_path = create_result.worktree.path

        # Should succeed - remove_worktree unlocks first
        remove_result = await manager.remove_worktree(worktree_path)
        assert remove_result.success is True


# --- Lock/Unlock Tests ---


class TestLockUnlock:
    """Tests for worktree locking operations."""

    @pytest.mark.asyncio
    async def test_lock_worktree(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test locking a worktree."""
        create_result = await manager.create_worktree(job_id="to-lock", lock=False)
        assert create_result.worktree is not None
        worktree_path = create_result.worktree.path

        lock_result = await manager.lock_worktree(worktree_path, reason="Test lock")
        assert lock_result.success is True

        # Verify locked
        info = await manager.get_worktree_info(worktree_path)
        assert info is not None
        assert info.locked is True

    @pytest.mark.asyncio
    async def test_lock_already_locked_idempotent(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test locking already locked worktree succeeds (idempotent)."""
        create_result = await manager.create_worktree(job_id="double-lock", lock=True)
        assert create_result.worktree is not None
        worktree_path = create_result.worktree.path

        # Lock again should succeed
        lock_result = await manager.lock_worktree(worktree_path)
        assert lock_result.success is True

    @pytest.mark.asyncio
    async def test_unlock_worktree(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test unlocking a worktree."""
        create_result = await manager.create_worktree(job_id="to-unlock", lock=True)
        assert create_result.worktree is not None
        worktree_path = create_result.worktree.path

        unlock_result = await manager.unlock_worktree(worktree_path)
        assert unlock_result.success is True

        # Verify unlocked
        info = await manager.get_worktree_info(worktree_path)
        assert info is not None
        assert info.locked is False

    @pytest.mark.asyncio
    async def test_unlock_not_locked_idempotent(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test unlocking not-locked worktree succeeds (idempotent)."""
        create_result = await manager.create_worktree(job_id="not-locked", lock=False)
        assert create_result.worktree is not None
        worktree_path = create_result.worktree.path

        # Unlock should succeed
        unlock_result = await manager.unlock_worktree(worktree_path)
        assert unlock_result.success is True


# --- List Worktrees Tests ---


class TestListWorktrees:
    """Tests for GitWorktreeManager.list_worktrees()."""

    @pytest.mark.asyncio
    async def test_list_empty(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test listing worktrees when none exist (except main)."""
        # With prefix filter, should be empty
        worktrees = await manager.list_worktrees(prefix_filter="mozart")
        assert len(worktrees) == 0

    @pytest.mark.asyncio
    async def test_list_with_worktrees(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test listing worktrees after creation."""
        await manager.create_worktree(job_id="list-test-1", lock=False)
        await manager.create_worktree(job_id="list-test-2", lock=False)

        worktrees = await manager.list_worktrees(prefix_filter="mozart")
        assert len(worktrees) == 2

        job_ids = {wt.job_id for wt in worktrees}
        assert "list-test-1" in job_ids
        assert "list-test-2" in job_ids

    @pytest.mark.asyncio
    async def test_list_with_prefix_filter(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test filtering worktrees by branch prefix."""
        await manager.create_worktree(job_id="mozart-1", branch_prefix="mozart", lock=False)
        await manager.create_worktree(job_id="other-1", branch_prefix="other", lock=False)

        mozart_worktrees = await manager.list_worktrees(prefix_filter="mozart")
        assert len(mozart_worktrees) == 1
        assert mozart_worktrees[0].job_id == "mozart-1"

        other_worktrees = await manager.list_worktrees(prefix_filter="other")
        assert len(other_worktrees) == 1
        assert other_worktrees[0].job_id == "other-1"


# --- Get Worktree Info Tests ---


class TestGetWorktreeInfo:
    """Tests for GitWorktreeManager.get_worktree_info()."""

    @pytest.mark.asyncio
    async def test_get_existing(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test getting info for existing worktree."""
        create_result = await manager.create_worktree(job_id="info-test", lock=False)
        assert create_result.worktree is not None
        worktree_path = create_result.worktree.path

        info = await manager.get_worktree_info(worktree_path)
        assert info is not None
        assert info.job_id == "info-test"
        assert info.branch == "mozart/info-test"

    @pytest.mark.asyncio
    async def test_get_nonexistent(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test getting info for nonexistent worktree returns None."""
        nonexistent = temp_git_repo / "nonexistent"
        info = await manager.get_worktree_info(nonexistent)
        assert info is None


# --- Prune Orphaned Tests ---


class TestPruneOrphaned:
    """Tests for GitWorktreeManager.prune_orphaned()."""

    @pytest.mark.asyncio
    async def test_prune_no_orphans(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test pruning when no orphans exist."""
        orphaned = await manager.prune_orphaned()
        assert orphaned == []

    @pytest.mark.asyncio
    async def test_prune_dry_run(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test dry run doesn't actually prune."""
        # Create and manually delete worktree directory
        create_result = await manager.create_worktree(job_id="orphan-test", lock=False)
        assert create_result.worktree is not None
        worktree_path = create_result.worktree.path

        # Manually remove directory (orphan it)
        import shutil
        shutil.rmtree(worktree_path)

        # Dry run should report but not prune
        orphaned = await manager.prune_orphaned(dry_run=True)
        assert len(orphaned) == 1

    @pytest.mark.asyncio
    async def test_prune_prefix_filter(
        self, manager: GitWorktreeManager, temp_git_repo: Path
    ) -> None:
        """Test pruning respects prefix filter."""
        await manager.create_worktree(job_id="keep-1", branch_prefix="other", lock=False)

        # Only prune mozart/* - should find nothing
        orphaned = await manager.prune_orphaned(prefix_filter="mozart")
        assert orphaned == []


# --- Helper Method Tests ---


class TestExtractJobId:
    """Tests for _extract_job_id helper."""

    def test_extract_with_prefix(self, manager: GitWorktreeManager) -> None:
        """Test extracting job ID from branch with prefix."""
        assert manager._extract_job_id("mozart/job-123") == "job-123"
        assert manager._extract_job_id("custom/my-job") == "my-job"

    def test_extract_without_prefix(self, manager: GitWorktreeManager) -> None:
        """Test extracting job ID from branch without prefix."""
        assert manager._extract_job_id("main") == "main"
        assert manager._extract_job_id("feature-branch") == "feature-branch"

    def test_extract_nested_prefix(self, manager: GitWorktreeManager) -> None:
        """Test extracting job ID from branch with nested slashes."""
        assert manager._extract_job_id("mozart/feature/sub") == "feature/sub"


# --- IsolationConfig Tests ---


class TestIsolationConfig:
    """Tests for IsolationConfig in config.py."""

    def test_default_config(self) -> None:
        """Test default IsolationConfig values."""
        from mozart.core.config import IsolationConfig, IsolationMode

        config = IsolationConfig()
        assert config.enabled is False
        assert config.mode == IsolationMode.WORKTREE
        assert config.branch_prefix == "mozart"
        assert config.cleanup_on_success is True
        assert config.cleanup_on_failure is False
        assert config.fallback_on_error is True

    def test_get_worktree_base_default(self, tmp_path: Path) -> None:
        """Test default worktree base path."""
        from mozart.core.config import IsolationConfig

        config = IsolationConfig()
        workspace = tmp_path / "workspace"
        assert config.get_worktree_base(workspace) == workspace / ".worktrees"

    def test_get_worktree_base_custom(self, tmp_path: Path) -> None:
        """Test custom worktree base path."""
        from mozart.core.config import IsolationConfig

        custom_base = tmp_path / "custom"
        config = IsolationConfig(worktree_base=custom_base)
        workspace = tmp_path / "workspace"
        assert config.get_worktree_base(workspace) == custom_base

    def test_get_branch_name(self) -> None:
        """Test branch name generation."""
        from mozart.core.config import IsolationConfig

        config = IsolationConfig(branch_prefix="myprefix")
        assert config.get_branch_name("job-123") == "myprefix/job-123"

    def test_branch_prefix_validation(self) -> None:
        """Test branch prefix pattern validation."""
        from mozart.core.config import IsolationConfig
        from pydantic import ValidationError

        # Valid prefixes
        IsolationConfig(branch_prefix="mozart")
        IsolationConfig(branch_prefix="my-prefix")
        IsolationConfig(branch_prefix="prefix_123")

        # Invalid prefixes (must start with letter)
        with pytest.raises(ValidationError):
            IsolationConfig(branch_prefix="123prefix")
        with pytest.raises(ValidationError):
            IsolationConfig(branch_prefix="-prefix")


class TestJobConfigIsolation:
    """Tests for isolation in JobConfig."""

    def test_job_config_has_isolation(self) -> None:
        """Test JobConfig has isolation field with default."""
        from mozart.core.config import IsolationConfig, JobConfig

        config_yaml = """
name: test-job
sheet:
  size: 10
  total_items: 100
prompt:
  template: "Test prompt"
"""
        import yaml
        data = yaml.safe_load(config_yaml)
        job_config = JobConfig.model_validate(data)

        assert hasattr(job_config, "isolation")
        assert isinstance(job_config.isolation, IsolationConfig)
        assert job_config.isolation.enabled is False

    def test_job_config_isolation_enabled(self) -> None:
        """Test JobConfig with isolation enabled."""
        from mozart.core.config import IsolationMode, JobConfig

        config_yaml = """
name: isolated-job
sheet:
  size: 10
  total_items: 100
prompt:
  template: "Test prompt"
isolation:
  enabled: true
  mode: worktree
  branch_prefix: test
"""
        import yaml
        data = yaml.safe_load(config_yaml)
        job_config = JobConfig.model_validate(data)

        assert job_config.isolation.enabled is True
        assert job_config.isolation.mode == IsolationMode.WORKTREE
        assert job_config.isolation.branch_prefix == "test"
